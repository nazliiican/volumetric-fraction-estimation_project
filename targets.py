"""Building the prediction targets from the metadata.

For every mixture (Code) we build three aligned vectors over the
material vocabulary:

    presence        -- multi-hot, 1 if the material is in the mixture
    mass_fraction   -- mass fraction of each material (sums to 1)
    volume_fraction -- volume fraction of each material (sums to 1)

Mass fractions come from the measured ``Weight`` column (the ground truth);
Volume fractions are derived from mass fractions and the per-material densities.
"""
import numpy as np

TOL = 1e-6


def parse_materials(cell):
    
    items = [x.strip() for x in str(cell).split(",")]
    items = [x for x in items if x != ""]
    if not items:
        raise ValueError(f"Could not parse any material from cell {cell!r}")
    return items


def parse_numeric_list(cell, expected_n):

    tokens = [t.strip() for t in str(cell).split(",")]
    values = [float(t) for t in tokens]
    if len(values) != expected_n:
        raise ValueError(
            f"Expected {expected_n} value(s) in cell {cell!r}, got {len(values)}"
        )
    return values


def build_material_vocabulary(metadata_df):

    vocab = set()
    for cell in metadata_df["Material"]:
        vocab.update(parse_materials(cell))
    return sorted(vocab)


def _build_density_vector(metadata_df, materials):

    density_by_material = {}
    for _, row in metadata_df.iterrows():
        names = parse_materials(row["Material"])
        densities = parse_numeric_list(row["Density"], len(names))
        for name, dens in zip(names, densities):
            if name in density_by_material:
                if abs(density_by_material[name] - dens) > TOL:
                    raise ValueError(
                        f"Inconsistent density for {name}: "
                        f"{density_by_material[name]} vs {dens}"
                    )
            else:
                density_by_material[name] = dens

    return np.array([density_by_material[m] for m in materials], dtype=float)


def mass_to_volume(mass_fraction, densities):

    mass_fraction = np.asarray(mass_fraction, dtype=float)
    densities = np.asarray(densities, dtype=float)
    volume = mass_fraction / densities
    total = volume.sum(axis=-1, keepdims=True)
    return volume / total


def volume_to_mass(volume_fraction, densities):

    volume_fraction = np.asarray(volume_fraction, dtype=float)
    densities = np.asarray(densities, dtype=float)
    mass = volume_fraction * densities
    total = mass.sum(axis=-1, keepdims=True)
    return mass / total


def build_targets(metadata_df, vocabulary=None):

    if vocabulary is None:
        vocabulary = build_material_vocabulary(metadata_df)
    index = {m: i for i, m in enumerate(vocabulary)}

    density_vector = _build_density_vector(metadata_df, vocabulary)

    codes = []
    n_codes = len(metadata_df)
    n_materials = len(vocabulary)

    presence = np.zeros((n_codes, n_materials), dtype=float)
    mass_fraction = np.zeros((n_codes, n_materials), dtype=float)

    for i, (_, row) in enumerate(metadata_df.iterrows()):
        code = row["Code"]
        codes.append(code)

        names = parse_materials(row["Material"])
        n = len(names)
        weights = parse_numeric_list(row["Weight"], n)
        total_weight = parse_numeric_list(row["Total weight"], 1)[0]

        percentages = parse_numeric_list(row["Percentage"], n)

        weights = np.array(weights, dtype=float)
        if np.any(weights < 0):
            raise ValueError(f"Negative weight in Code {code}: {weights.tolist()}")

        if abs(weights.sum() - total_weight) > 1.0:  
            raise ValueError(
                f"Code {code}: weights sum to {weights.sum()} but "
                f"Total weight is {total_weight}"
            )
        frac = weights / weights.sum()

        for name, f in zip(names, frac):
            if name not in index:
                raise ValueError(f"Material {name!r} (Code {code}) not in vocabulary")
            presence[i, index[name]] = 1.0
            mass_fraction[i, index[name]] = f

        pct_frac = np.array(percentages, dtype=float) / np.sum(percentages)
        if np.max(np.abs(pct_frac - frac)) > 0.05:
            raise ValueError(
                f"Code {code}: Percentage and Weight disagree "
                f"({percentages} vs weights {weights.tolist()})"
            )

    density_matrix = np.tile(density_vector, (n_codes, 1))
    volume_fraction = mass_to_volume(mass_fraction, density_vector)

    return {
        "codes": codes,
        "materials": list(vocabulary),
        "presence": presence,
        "mass_fraction": mass_fraction,
        "volume_fraction": volume_fraction,
        "density_matrix": density_matrix,
    }


def validate_targets(targets):
    
    presence = np.asarray(targets["presence"])
    mass = np.asarray(targets["mass_fraction"])
    volume = np.asarray(targets["volume_fraction"])

    assert presence.shape == mass.shape == volume.shape, "Target shapes differ"

    assert np.all(np.isin(presence, [0.0, 1.0])), "presence must be 0/1 only"

    assert np.all(mass >= -TOL), "mass_fraction has negative entries"
    assert np.all(volume >= -TOL), "volume_fraction has negative entries"

    mass_sums = mass.sum(axis=1)
    vol_sums = volume.sum(axis=1)
    assert np.allclose(mass_sums, 1.0, atol=TOL), (
        f"mass_fraction rows do not sum to 1 (min={mass_sums.min()}, "
        f"max={mass_sums.max()})"
    )
    assert np.allclose(vol_sums, 1.0, atol=TOL), (
        f"volume_fraction rows do not sum to 1 (min={vol_sums.min()}, "
        f"max={vol_sums.max()})"
    )

    mass_support = (mass > TOL).astype(float)
    vol_support = (volume > TOL).astype(float)
    assert np.array_equal(mass_support, presence), (
        "mass_fraction support does not match presence"
    )
    assert np.array_equal(vol_support, presence), (
        "volume_fraction support does not match presence"
    )

    return True
