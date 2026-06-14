"""Grouped train/val/test splitting
Every split is grouped by ``Code``. The 5 images of a mixture show the same
physical composition, so splitting at the image level would leak information
between train and test. These guarantee that all 5 images of a Code
always stay together in a single split, and assert that no leakage occurs.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def grouped_train_val_test_split(
    df, group_col="Code", test_size=0.2, val_size=0.15, seed=42
):
    
    groups = df[group_col].values

    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval_idx, test_idx = next(gss_test.split(df, groups=groups))
    trainval = df.iloc[trainval_idx]
    test = df.iloc[test_idx]

    val_relative = val_size / (1.0 - test_size)
    gss_val = GroupShuffleSplit(
        n_splits=1, test_size=val_relative, random_state=seed
    )
    train_idx, val_idx = next(
        gss_val.split(trainval, groups=trainval[group_col].values)
    )
    train = trainval.iloc[train_idx]
    val = trainval.iloc[val_idx]

    check_no_leakage(train, val, test, group_col=group_col)

    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def grouped_kfold_split(df, group_col="Code", n_splits=5, seed=42):
    unique_groups = np.array(sorted(pd.unique(df[group_col])))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_groups)

    folds = np.array_split(unique_groups, n_splits)

    splits = []
    for val_groups in folds:
        val_mask = df[group_col].isin(set(val_groups.tolist()))
        val_df = df[val_mask].reset_index(drop=True)
        train_df = df[~val_mask].reset_index(drop=True)
        check_no_leakage(train_df, val_df, group_col=group_col)
        splits.append((train_df, val_df))
    return splits


def check_no_leakage(*splits, group_col="Code"):
   
    group_sets = [set(s[group_col].unique()) for s in splits]
    for i in range(len(group_sets)):
        for j in range(i + 1, len(group_sets)):
            shared = group_sets[i] & group_sets[j]
            assert not shared, (
                f"Leakage: groups {sorted(shared)} appear in both split {i} "
                f"and split {j}"
            )
    return True


def check_groups_intact(splits, full_df, group_col="Code"):

    expected_counts = full_df.groupby(group_col).size().to_dict()

    assigned = {}  
    for i, s in enumerate(splits):
        for group, count in s.groupby(group_col).size().items():
            assert group not in assigned, (
                f"Group {group} appears in multiple splits "
                f"({assigned[group][0]} and {i})"
            )
            assigned[group] = (i, count)

    for group, full_count in expected_counts.items():
        assert group in assigned, f"Group {group} is missing from all splits"
        _, count = assigned[group]
        assert count == full_count, (
            f"Group {group} is split across folds: {count} of {full_count} rows"
        )

    total_rows = sum(len(s) for s in splits)
    assert total_rows == len(full_df), (
        f"Row count mismatch: splits have {total_rows}, full has {len(full_df)}"
    )
    return True
