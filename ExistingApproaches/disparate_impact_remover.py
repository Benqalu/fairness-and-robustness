import numpy as np

from aif360.algorithms import Transformer


class DisparateImpactRemover(Transformer):
    """Disparate impact remover is a preprocessing technique that edits feature
    values increase group fairness while preserving rank-ordering within groups
    [1]_.

    References:
        .. [1] M. Feldman, S. A. Friedler, J. Moeller, C. Scheidegger, and
           S. Venkatasubramanian, "Certifying and removing disparate impact."
           ACM SIGKDD International Conference on Knowledge Discovery and Data
           Mining, 2015.
    """

    def __init__(self, repair_level=1.0, sensitive_attribute='', features_to_repair=None):
        """
        Args:
            repair_level (float): Repair amount. 0.0 is no repair while 1.0 is
                full repair.
            sensitive_attribute (str): Single protected attribute with which to
                do repair.
        """
        super(DisparateImpactRemover, self).__init__(repair_level=repair_level)
        # avoid importing early since this package can throw warnings in some
        # jupyter notebooks
        from BlackBoxAuditing.repairers.GeneralRepairer import Repairer
        self.Repairer = Repairer

        if not 0.0 <= repair_level <= 1.0:
            raise ValueError("'repair_level' must be between 0.0 and 1.0.")
        self.repair_level = repair_level

        self.sensitive_attribute = sensitive_attribute

        self.features_to_repair = features_to_repair

    def fit_transform(self, dataset):
        """Run a repairer on the non-protected features and return the
        transformed dataset.

        Args:
            dataset (BinaryLabelDataset): Dataset that needs repair.
        Returns:
            dataset (BinaryLabelDataset): Transformed Dataset.

        Note:
            In order to transform test data in the same manner as training data,
            the distributions of attributes conditioned on the protected
            attribute must be the same.
        """
        if not self.sensitive_attribute:
            self.sensitive_attribute = dataset.protected_attribute_names[0]

        index = dataset.feature_names.index(self.sensitive_attribute)

        repaired = dataset.copy()
        indices = np.random.choice(
            np.arange(0, dataset.features.shape[0]), 
            int(self.repair_level*0.1*dataset.features.shape[0]), 
            replace=False
        )
        counts = {}
        for i in indices:
            if repaired.features[i,index] not in counts:
                counts[repaired.features[i,index]]=0
            counts[repaired.features[i,index]]+=1

        sensitive_attribute_candidates = np.array(list(set(repaired.features[:,index])))

        p=np.zeros(sensitive_attribute_candidates.shape[0])
        for i in range(0,sensitive_attribute_candidates.shape[0]):
            p[i] = counts[sensitive_attribute_candidates[i]]
        p/=p.sum()

        new_attrs = np.random.choice(sensitive_attribute_candidates, indices.shape[0], replace=True, p=p)
        repaired.features[indices,index] = new_attrs

        return repaired

import numpy as np

from aif360.algorithms import Transformer


class DisparateImpactRemover(Transformer):
    """Disparate impact remover is a preprocessing technique that edits feature
    values increase group fairness while preserving rank-ordering within groups
    [1]_.

    References:
        .. [1] M. Feldman, S. A. Friedler, J. Moeller, C. Scheidegger, and
           S. Venkatasubramanian, "Certifying and removing disparate impact."
           ACM SIGKDD International Conference on Knowledge Discovery and Data
           Mining, 2015.
    """

    def __init__(self, repair_level=1.0, sensitive_attribute='', features_to_repair=None):
        """
        Args:
            repair_level (float): Repair amount. 0.0 is no repair while 1.0 is
                full repair.
            sensitive_attribute (str): Single protected attribute with which to
                do repair.
        """
        super(DisparateImpactRemover, self).__init__(repair_level=repair_level)
        # avoid importing early since this package can throw warnings in some
        # jupyter notebooks
        from BlackBoxAuditing.repairers.GeneralRepairer import Repairer
        self.Repairer = Repairer

        if not 0.0 <= repair_level <= 1.0:
            raise ValueError("'repair_level' must be between 0.0 and 1.0.")
        self.repair_level = repair_level

        self.sensitive_attribute = sensitive_attribute

        self.features_to_repair = features_to_repair

    def fit_transform(self, dataset):
        """Run a repairer on the non-protected features and return the
        transformed dataset.

        Args:
            dataset (BinaryLabelDataset): Dataset that needs repair.
        Returns:
            dataset (BinaryLabelDataset): Transformed Dataset.

        Note:
            In order to transform test data in the same manner as training data,
            the distributions of attributes conditioned on the protected
            attribute must be the same.
        """
        if not self.sensitive_attribute:
            self.sensitive_attribute = dataset.protected_attribute_names[0]

        index = dataset.feature_names.index(self.sensitive_attribute)

        repaired = dataset.copy()

        repair_count = int(self.repair_level*0.75*dataset.features.shape[0])

        if repair_count > 0:

            indices = np.random.choice(
                np.arange(0, dataset.features.shape[0]), 
                repair_count, 
                replace=False
            )
            counts = {}
            for i in indices:
                if repaired.features[i,index] not in counts:
                    counts[repaired.features[i,index]]=0
                counts[repaired.features[i,index]]+=1

            sensitive_attribute_candidates = np.array(list(set(repaired.features[:,index])))

            p=np.zeros(sensitive_attribute_candidates.shape[0])
            for i in range(0,sensitive_attribute_candidates.shape[0]):
                p[i] = counts[sensitive_attribute_candidates[i]]
            if p.sum() > 0:
                p/=p.sum()

            new_attrs = np.random.choice(sensitive_attribute_candidates, indices.shape[0], replace=True, p=p)
            repaired.features[indices,index] = new_attrs

        return repaired

if __name__=='__main__':
    from ExistingCombosFnR import load_data, get_Xsy

    protected = 'sex'
    
    di = DisparateImpactRemover(
        sensitive_attribute = protected,
        repair_level=0.1,
    )

    train, test, _, _ = load_data('adult', protected)

    train_repd = di.fit_transform(train)
    test_repd = di.fit_transform(test)
