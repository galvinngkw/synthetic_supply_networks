# OECD input-output tables

This directory contains country-level input-output tables used by the project. The tables are derived from the [OECD Inter-Country Input-Output (ICIO) 2021 release](https://www.oecd.org/en/topics/sub-issues/input-output-tables.html) and are stored as Python pickle files.

## Files

Each file is named `{COUNTRY_CODE}.pkl`, where `{COUNTRY_CODE}` is the country or economy code used by the project, for example `HUN.pkl` or `BEL.pkl`. The repository currently includes files for the country and economy codes present in this directory.

Each pickle is expected to contain a mapping of years to square industry-by-industry input-output matrices. Values are in millions of US dollars unless otherwise specified by the source data. The loader applies the configured monetary-unit scaling when the table is used by the network generator.

The default configuration loads Hungary (`HUN`) for 2015. To use another table or year, update `country_code` and `year` in `conf/network/default.yaml`, or pass those values through `NetworkConfig`.

## Loading

The project loads these files through `network_generator.data_iot.load_input_output_table`:

```python
from network_generator.data_iot import load_input_output_table

input_output_tensor, input_output_array, num_industries = load_input_output_table(
    data_path="data",
    country_code="HUN",
    year=2015,
    monetary_unit=1.0e8,
)
```

The exact function arguments and scaling behavior are defined by the current implementation and configuration.

## Terms and attribution

The MIT License at the repository root covers the project software, not these data files. Data use is governed by the [OECD Terms & Conditions](https://www.oecd.org/en/about/terms-conditions.html), including its provisions concerning Data, and the [OECD Open Access Policy](https://www.oecd.org/en/about/oecd-open-by-default-policy.html). Check the original source metadata for third-party ownership, additional restrictions, and any source-specific citation requirements before reusing or redistributing a table.

Please cite the OECD ICIO 2021 release and state that the source tables were converted to Python pickle format for this repository. Use of the data or project does not imply OECD endorsement.

See [`../DATA_NOTICE.md`](../DATA_NOTICE.md) for the full data notice.
