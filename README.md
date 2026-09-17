# Healthcare Drug Safety Analysis (openFDA)

Analysis of FDA Adverse Event Reporting System (FAERS) data, pulled live via the [openFDA API](https://open.fda.gov/apis/drug/event/), comparing adverse event patterns across three widely-used drugs: **metformin** (diabetes), **ibuprofen** (pain/anti-inflammatory), and **atorvastatin** (cholesterol).

## Analysis Questions

1. **Which drug has the highest rate of serious adverse events**, and how does that break down by outcome type -> death, life-threatening, hospitalization, disabling, congenital anomaly?
2. **Does administration route predict severity?** e.g. is an oral drug's serious-event rate different from the same drug taken another way.
3. **What's the reaction outcome distribution per drug** -> recovered, recovering, not recovered, recovered with sequelae, fatal, unknown? A severity gradient, not just a binary flag.
4. **What are the most common reaction terms per drug**, and do the top reactions differ meaningfully between the three drugs?

5. **Geographic pattern** -> which countries report the most events per drug?
6. **Age and sex patterns** -> do serious events skew toward particular age groups or a particular sex, and does this differ by drug?
7. **Multi-drug reports** -> how often does a report list more than one drug, and does that complicate attributing a reaction to a single drug?

## Data Source

- FDA Adverse Event Reporting System (FAERS), via openFDA

## Dataset

Output of `data_fetch.py`, saved to `data/raw_adverse_events.csv`. Each row represents one adverse reaction reported against one drug within one FAERS safety report. so a single report with 3 reactions produces 3 rows, linked by `safetyreportid`.

| Column                         | Type          | Description                                                                                                                               |
| ------------------------------ | ------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `safetyreportid`               | string        | Unique FAERS report ID. Links rows back to the same original report.                                                                      |
| `queried_drug`                 | string        | Which of the three drugs this record was fetched for (metformin/ibuprofen/atorvastatin).                                                  |
| `drug_names_on_report`         | string        | All drug names listed on the report, `;`-separated. Often more than one — see the multi-drug-report question above.                       |
| `drug_routes_on_report`        | string        | Administration route(s) for the drug(s) on the report (oral, IV, topical, etc.), `;`-separated.                                           |
| `reaction_term`                | string        | The adverse reaction, as a standardized MedDRA term (e.g. "Nausea", "Hepatic failure").                                                   |
| `reaction_outcome`             | string (code) | Outcome of this specific reaction: `1`=recovered, `2`=recovering, `3`=not recovered, `4`=recovered with sequelae, `5`=fatal, `6`=unknown. |
| `serious`                      | string (code) | `1` = report met at least one FDA seriousness criterion, `2` = not serious.                                                               |
| `seriousnessdeath`             | string (flag) | `1` if death was an outcome, else `0`.                                                                                                    |
| `seriousnesshospitalization`   | string (flag) | `1` if hospitalization occurred, else `0`.                                                                                                |
| `seriousnesslifethreatening`   | string (flag) | `1` if life-threatening, else `0`.                                                                                                        |
| `seriousnessdisabling`         | string (flag) | `1` if resulted in disability, else `0`.                                                                                                  |
| `seriousnesscongenitalanomali` | string (flag) | `1` if a congenital anomaly/birth defect, else `0`.                                                                                       |
| `occurcountry`                 | string        | ISO country code where the event occurred.                                                                                                |
| `patient_sex`                  | string (code) | `1`=male, `2`=female, unreported values may be blank.                                                                                     |
| `patient_age`                  | string/float  | Patient age at event onset. Unit not captured in this version — mostly years, but check for outliers before analysis.                     |
| `receivedate`                  | string (date) | Date FDA received the report, format `YYYYMMDD`.                                                                                          |
