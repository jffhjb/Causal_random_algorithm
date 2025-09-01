
"""
Module to generate diverse counterfactual explanations based on random sampling.
A simple implementation.
"""
import random
import timeit

import numpy as np
import pandas as pd

from dice_ml import diverse_counterfactuals as exp
from dice_ml.constants import ModelTypes
from dice_ml.explainer_interfaces.explainer_base import ExplainerBase
from dowhy import gcm

class CausalRandom(ExplainerBase):

    def __init__(self, data_interface, model_interface, gcm_model):
        """Init method

        :param data_interface: an interface class to access data related params.
        :param model_interface: an interface class to access trained ML model.
        """
        super().__init__(data_interface)  # initiating data related parameters

        self.model = model_interface
        self.model.load_model()  # loading pickled trained model if applicable
        self.model.transformer.feed_data_params(data_interface)
        self.model.transformer.initialize_transform_func()
        self.gcm_model = gcm_model  # 🟢 Add this line

        self.precisions = self.data_interface.get_decimal_precisions(output_type="dict")
        if self.data_interface.outcome_name in self.precisions:
            self.outcome_precision = [self.precisions[self.data_interface.outcome_name]]
        else:
            self.outcome_precision = 0

    def _generate_counterfactuals(self, query_instance, total_CFs, desired_range=None,
                                  desired_class="opposite", permitted_range=None,
                                  features_to_vary="all", stopping_threshold=0.5, posthoc_sparsity_param=0.1,
                                  posthoc_sparsity_algorithm="linear", sample_size=1000, random_seed=None, verbose=False,
                                  limit_steps_ls=10000):
        """Generate counterfactuals by randomly sampling features.

        :param query_instance: Test point of interest. A dictionary of feature names and values or a single row dataframe.
        :param total_CFs: Total number of counterfactuals required.
        :param desired_range: For regression problems. Contains the outcome range to generate counterfactuals in.
        :param desired_class: Desired counterfactual class - can take 0 or 1. Default value is "opposite" to the outcome
                              class of query_instance for binary classification.
        :param permitted_range: Dictionary with feature names as keys and permitted range in list as values.
                                Defaults to the range inferred from training data. If None, uses the parameters
                                initialized in data_interface.
        :param features_to_vary: Either a string "all" or a list of feature names to vary.
        :param stopping_threshold: Minimum threshold for counterfactuals target class probability.
        :param posthoc_sparsity_param: Parameter for the post-hoc operation on continuous features to enhance sparsity.
        :param posthoc_sparsity_algorithm: Perform either linear or binary search. Takes "linear" or "binary".
                                           Prefer binary search when a feature range is large
                                           (for instance, income varying from 10k to 1000k) and only if the features
                                           share a monotonic relationship with predicted outcome in the model.
        :param sample_size: Sampling size
        :param random_seed: Random seed for reproducibility
        :param limit_steps_ls: Defines an upper limit for the linear search step in the posthoc_sparsity_enhancement

        :returns: A CounterfactualExamples object that contains the dataframe of generated counterfactuals as an attribute.
        """
        # Set features to vary
        self.features_to_vary = self.setup(features_to_vary, permitted_range, query_instance, feature_weights=None)

        if features_to_vary == "all":
            self.fixed_features_values = {}
        else:
            self.fixed_features_values = {}
            for feature in self.data_interface.feature_names:
                if feature not in features_to_vary:
                    self.fixed_features_values[feature] = query_instance[feature].iat[0]

        # Do predictions once on the query_instance and reuse across to reduce the number
        # inferences.
        model_predictions = self.predict_fn(query_instance)
        # number of output nodes of ML model
        self.num_output_nodes = None
        if self.model.model_type == ModelTypes.Classifier:
            self.num_output_nodes = model_predictions.shape[1]

        # query_instance need no transformation for generating CFs using random sampling.
        # find the predicted value of query_instance
        test_pred = model_predictions[0]
        if self.model.model_type == ModelTypes.Classifier:
            self.target_cf_class = self.infer_target_cfs_class(desired_class, test_pred, self.num_output_nodes)
        elif self.model.model_type == ModelTypes.Regressor:
            self.target_cf_range = self.infer_target_cfs_range(desired_range)
        # fixing features that are to be fixed
        self.total_CFs = total_CFs

        self.stopping_threshold = stopping_threshold
        if self.model.model_type == ModelTypes.Classifier:
            # TODO Generalize this for multi-class
            if self.target_cf_class == 0 and self.stopping_threshold > 0.5:
                self.stopping_threshold = 0.25
            elif self.target_cf_class == 1 and self.stopping_threshold < 0.5:
                self.stopping_threshold = 0.75

        # get random samples for each feature independently
        start_time = timeit.default_timer()
        
        # Calculate feature ranges based on 5th to 95th percentiles
        safe_feature_range = {}
        for feature in self.data_interface.feature_names:
            if feature in self.data_interface.continuous_feature_names:
                # For continuous features, use 5th to 95th percentiles
                series = self.data_interface.data_df[feature]
                try:
                    low_q = float(np.nanpercentile(series.values, 5))
                    high_q = float(np.nanpercentile(series.values, 95))
                    if not np.isfinite(low_q) or not np.isfinite(high_q) or low_q >= high_q:
                        # If percentile calculation fails, fall back to original range
                        low_q = float(np.nanmin(series.values))
                        high_q = float(np.nanmax(series.values))
                except Exception:
                    # In case of exception, fall back to original range
                    low_q = float(np.nanmin(series.values))
                    high_q = float(np.nanmax(series.values))
                safe_feature_range[feature] = [low_q, high_q]
            else:
                # For categorical features, use original range
                safe_feature_range[feature] = list(self.feature_range[feature])
        
        # Generate random samples
        random_instances = self.get_samples(
            self.fixed_features_values,
            safe_feature_range, sampling_random_seed=random_seed, sampling_size=sample_size)
        # Generate copies of the query instance that will be changed one feature
        # at a time to encourage sparsity.
        cfs_df = None
        # Copy query_instance as counterfactual template
        candidate_cfs = pd.DataFrame(
            np.repeat(query_instance.values, sample_size, axis=0), columns=query_instance.columns)
        for col in candidate_cfs.columns:
            candidate_cfs[col] = candidate_cfs[col].astype(query_instance[col].dtype)


        intervention_history = np.empty((sample_size, len(self.features_to_vary)), dtype=object)

        # Loop to change one feature at a time, then two features, and so on.
        for num_features_to_vary in range(1, len(self.features_to_vary)+1):

            # Randomly select one feature
            selected_features = np.random.choice(self.features_to_vary, (sample_size, 1), replace=True)

            # Replace random samples into counterfactual template
            for k in range(sample_size):
                
                feature = selected_features[k][0] 

                # Insert into gcm.counterfactual_samples call
                candidate_cfs.loc[candidate_cfs.index[k], :] = gcm.counterfactual_samples(
                    self.gcm_model,
                    {feature: lambda x: random_instances.at[k, feature]},
                    observed_data=candidate_cfs.iloc[[k]]
                )[self.data_interface.feature_names].iloc[0].values

                intervention_history[k, num_features_to_vary - 1] = (feature, random_instances.at[k, feature])



                # Fix cases where some variables in candidate_cfs exceed the range
            for feature in self.data_interface.categorical_feature_names:
                min_val = self.data_interface.data_df[feature].cat.codes.min()
                max_val = self.data_interface.data_df[feature].cat.codes.max()
                candidate_cfs[feature] = candidate_cfs[feature].astype(int).clip(lower=min_val,
                                                                                upper=max_val)
            

            

            # Predict counterfactual samples
            scores = self.predict_fn(candidate_cfs)
            # Determine if counterfactual samples are valid
            validity = self.decide_cf_validity(scores)

            
            # If counterfactual samples are valid, add them to counterfactual samples
            if sum(validity) > 0:
                rows_to_add = candidate_cfs[validity == 1]
                if cfs_df is None:
                    cfs_df = rows_to_add.copy()
                else:
                    cfs_df = pd.concat([cfs_df, rows_to_add])
                cfs_df.drop_duplicates(inplace=True)
                
                # Always change at least 2 features before stopping
                if num_features_to_vary >= 2 and len(cfs_df) >= total_CFs:
                    break

        
        self.total_cfs_found = 0
        self.valid_cfs_found = False
        if cfs_df is not None and len(cfs_df) > 0:
            if len(cfs_df) > total_CFs:
                cfs_df = cfs_df.sample(total_CFs)
            intervention_log = intervention_history[cfs_df.index, :num_features_to_vary]
            cfs_df.reset_index(inplace=True, drop=True)
            if len(cfs_df) > 0:
                self.cfs_pred_scores = self.predict_fn(cfs_df)
                cfs_df[self.data_interface.outcome_name] = self.get_model_output_from_scores(self.cfs_pred_scores)
            else:
                if self.model.model_type == ModelTypes.Classifier:
                    self.cfs_pred_scores = [0]*self.num_output_nodes
                else:
                    self.cfs_pred_scores = [0]
            self.total_cfs_found = len(cfs_df)

            self.valid_cfs_found = True if self.total_cfs_found >= self.total_CFs else False
            if len(cfs_df) > 0:
                final_cfs_df = cfs_df[self.data_interface.feature_names + [self.data_interface.outcome_name]]
                final_cfs_df[self.data_interface.outcome_name] = \
                    final_cfs_df[self.data_interface.outcome_name].round(self.outcome_precision)
                self.cfs_preds = final_cfs_df[[self.data_interface.outcome_name]].values
                self.final_cfs = final_cfs_df[self.data_interface.feature_names].values
            else:
                final_cfs_df = None
                self.cfs_preds = None
                self.cfs_pred_scores = None
                self.final_cfs = None
        else:
            final_cfs_df = None
            self.cfs_preds = None
            self.cfs_pred_scores = None
            self.final_cfs = None
        test_instance_df = self.data_interface.prepare_query_instance(query_instance)
        test_instance_df[self.data_interface.outcome_name] = \
            np.array(np.round(self.get_model_output_from_scores((test_pred,)), self.outcome_precision))
        # post-hoc operation on continuous features to enhance sparsity - only for public data
        if posthoc_sparsity_param is not None and posthoc_sparsity_param > 0 and \
                self.final_cfs is not None and 'data_df' in self.data_interface.__dict__:
            final_cfs_df_sparse = final_cfs_df.copy()
            final_cfs_df_sparse = self.do_posthoc_sparsity_enhancement(final_cfs_df_sparse,
                                                                       test_instance_df,
                                                                       posthoc_sparsity_param,
                                                                       posthoc_sparsity_algorithm,
                                                                       limit_steps_ls)
        elif self.final_cfs is not None:
            final_cfs_df_sparse = final_cfs_df.copy()
        else:
            final_cfs_df_sparse = None

        self.elapsed = timeit.default_timer() - start_time
        m, s = divmod(self.elapsed, 60)

        # decoding to original label
        test_instance_df, final_cfs_df, final_cfs_df_sparse = \
            self.decode_to_original_labels(test_instance_df, final_cfs_df, final_cfs_df_sparse)
        if final_cfs_df is not None:
            if verbose:
                print('Diverse Counterfactuals found! total time taken: %02d' %
                      m, 'min %02d' % s, 'sec')
        else:
            if self.total_cfs_found == 0:
                print('No Counterfactuals found for the given configuration, perhaps try with different parameters...',
                      '; total time taken: %02d' % m, 'min %02d' % s, 'sec')
            else:
                print('Only %d (required %d) ' % (self.total_cfs_found, self.total_CFs),
                      'Diverse Counterfactuals found for the given configuration, perhaps try with different parameters...',
                      '; total time taken: %02d' % m, 'min %02d' % s, 'sec')

        desired_class_param = self.decode_model_output(pd.Series(self.target_cf_class))[0] \
            if hasattr(self, 'target_cf_class') else desired_class
        
        
        cf_result = exp.CounterfactualExamples(
            data_interface=self.data_interface,
            final_cfs_df=final_cfs_df,
            test_instance_df=test_instance_df,
            final_cfs_df_sparse=final_cfs_df_sparse,
            posthoc_sparsity_param=posthoc_sparsity_param,
            desired_class=desired_class_param,
            desired_range=desired_range,
            model_type=self.model.model_type
        )

        # ✅ Dynamically attach intervention_history to CounterfactualExamples object
        cf_result.intervention_log = intervention_log

        return cf_result
       

    def get_samples(self, fixed_features_values, feature_range, sampling_random_seed, sampling_size):

        # first get required parameters
        precisions = self.data_interface.get_decimal_precisions(output_type="dict")

        if sampling_random_seed is not None:
            random.seed(sampling_random_seed)

        samples = []
        # Iterate through all features
        for feature in self.data_interface.feature_names:
            if feature in fixed_features_values:
                # If feature is in fixed feature values, use fixed feature value
                sample = [fixed_features_values[feature]]*sampling_size
            # If feature is continuous, generate continuous samples
            elif feature in self.data_interface.continuous_feature_names:
                # Get feature range
                low = feature_range[feature][0]
                high = feature_range[feature][1]
                sample = self.get_continuous_samples(
                    low, high, precisions[feature], size=sampling_size,
                    seed=sampling_random_seed)
            else:
                # If feature is categorical, generate categorical samples
                if sampling_random_seed is not None:
                    random.seed(sampling_random_seed)
                sample = random.choices(feature_range[feature], k=sampling_size)

            samples.append(sample)
        samples = pd.DataFrame(dict(zip(self.data_interface.feature_names, samples)))
        return samples

    def get_continuous_samples(self, low, high, precision, size=1000, seed=None):

        if seed is not None:
            np.random.seed(seed)

        # If precision is 0, generate integer samples
        if precision == 0:
            result = np.random.randint(low, high+1, size).tolist()
            result = [float(r) for r in result]
        else:
            # If precision is not 0, generate float samples
            result = np.random.uniform(low, high+(10**-precision), size)
            result = [round(r, precision) for r in result]
        return result
    
