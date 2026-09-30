# Healthcare Drug Safety Analysis (openFDA)

This project studies real adverse event reports from the FDA. Data comes
live from the [openFDA API](https://open.fda.gov/apis/drug/event/), for
three common drugs: metformin (diabetes), ibuprofen (pain relief), and
atorvastatin (cholesterol). The project has two parts: a data analysis, and
a machine learning model that predicts if a report is serious, using only
the text of the reaction.

## What is in this project

1. A script that fetches raw report data from the openFDA API
2. A notebook that cleans the data and answers six analysis questions
3. Scripts that build and compare two models: a zero shot model and a fine
   tuned model
4. Charts saved in the `results` folder

## Analysis Questions

1. Which drug has the most serious reports, and what kind (death, life
   threatening, hospital stay, disability, birth defect)?
2. Does the route the drug is given by (pill, patch, injection) relate to
   how serious the report is?
3. What happens after the reaction (recovered, not recovered, fatal,
   unknown)?
4. Which reactions are specific to one drug, not just common in general?
5. Which countries report the most events?
6. Do age and sex patterns differ between the three drugs?

## Data Source

FDA Adverse Event Reporting System (FAERS), through the openFDA API. This
is a public system where doctors, pharmacists, drug companies and patients
report side effects. It is voluntary. It shows what people chose to report,
not the true rate of harm from a drug.

## Dataset

The raw data comes from `data_fetch.py` and is saved to
`data/raw_adverse_events.csv`. One row is one reaction reported for one
drug within one report. A report with 3 reactions makes 3 rows, all sharing
the same `safetyreportid`.

| Column                         | Type        | Description                                                                                                                          |
| ------------------------------ | ----------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `safetyreportid`               | string      | The report ID. Links rows from the same report.                                                                                      |
| `queried_drug`                 | string      | Which of the three drugs we searched for when we got this row.                                                                       |
| `drug_names_on_report`         | string      | Every drug listed on the report, separated by `;`. Most reports list many drugs.                                                     |
| `drug_routes_on_report`        | string      | The route for each drug on the report (oral, IV, patch, and so on), separated by `;`. Some drugs have no route recorded.             |
| `reaction_term`                | string      | The reaction, as a standard medical term (example: Nausea).                                                                          |
| `reaction_outcome`             | number code | What happened after the reaction. 1 recovered, 2 recovering, 3 not recovered, 4 recovered with a lasting effect, 5 fatal, 6 unknown. |
| `serious`                      | number code | 1 means the report met at least one FDA seriousness rule. 2 means not serious.                                                       |
| `seriousnessdeath`             | flag        | 1 if death happened.                                                                                                                 |
| `seriousnesshospitalization`   | flag        | 1 if the patient went to hospital.                                                                                                   |
| `seriousnesslifethreatening`   | flag        | 1 if the event was life threatening.                                                                                                 |
| `seriousnessdisabling`         | flag        | 1 if it caused a disability.                                                                                                         |
| `seriousnesscongenitalanomali` | flag        | 1 if it caused a birth defect.                                                                                                       |
| `occurcountry`                 | string      | Country code where the event happened. Missing values are filled with the word Unknown.                                              |
| `patient_sex`                  | number code | 1 male, 2 female.                                                                                                                    |
| `patient_age`                  | number      | Age of the patient. Do not use this alone, see the note below.                                                                       |
| `patient_age_unit`             | number code | The unit for `patient_age`. 800 decade, 801 year, 802 month, 803 week, 804 day, 805 hour.                                            |
| `patient_age_years`            | number      | Age converted to years using the unit above. Use this column for any age analysis, not `patient_age`.                                |
| `receivedate`                  | date        | Date the FDA received the report.                                                                                                    |

The notebook also builds a few extra columns during cleaning:

- `drug_count` and `is_multi_drug`: how many different drugs are listed on
  the report
- `route_code`, `route_label`, `route_count`: the drug route, turned from a
  number into a plain word (example: 048 becomes Oral)
- `reaction_outcome_label`: the outcome code turned into a plain word
- `outcome_conflict`: True if the same reaction had two different outcomes
  recorded, in which case we treat the outcome as unknown rather than guess

### An important limit of this data

Almost every report lists many drugs (about 8 on average). A route or a
reaction found on a report may belong to a different drug on the same
report, not the one we searched for. So a finding often describes the
report as a whole, not one single drug for certain. This limit is repeated
in the notebook next to the results it affects most.

## Key Findings

**Question 1, type of serious outcome.** All three drugs have a similar
overall serious rate, near 80 percent. The differences show up in the type
of serious outcome. Metformin has the highest rate for most specific
outcomes, including death. This may reflect that people who take metformin
are often older and managing other health problems, not that the drug
itself is worse.

**Question 2, route of the drug.** Patches had a much lower serious rate
(about 20 percent) than any other route. IV had the highest (about 93
percent). Oral, the most common route, sat close to the overall average
(about 78 percent). This fits how each route puts the drug into the body.
An IV works fast and fully. A patch releases slowly through the skin.

**Question 3, what happens after the reaction.** For all three drugs, the
single most common answer is Unknown, near 35 to 39 percent. This means
most reports never say what happened next. Fatal outcomes are rare and
similar across the three drugs (about 2 to 3 percent).

**Question 4, reactions specific to one drug.** Comparing each drug against
the other two found real, known risks. Metformin showed lactic acidosis and
metabolic acidosis. Atorvastatin showed myalgia and rhabdomyolysis, the
known muscle damage risk of statins. Both metformin and ibuprofen also
showed a higher rate of suicide related reports. We do not say the drug
causes this. It more likely reflects that people with long term illness
already have a higher risk of this, for reasons apart from the drug.

**Question 5, which countries report the most.** The United States makes up
about 47 percent of all reports, and the UK about 14 percent. This is
because FAERS is a US system. It does not mean the drug is riskier in these
countries.

**Question 6, age and sex.** Ibuprofen skews much younger (middle age of
report is 50), which fits its common use for children. Metformin (64) and
atorvastatin (68) skew older, which fits who typically has diabetes and
high cholesterol.

## Model Comparison: Zero Shot vs Fine Tuned

A separate part of this project builds a small model that reads only the
reaction text and predicts if the report is serious.

![Model comparison](results/zero_shot_vs_fine_tuned.png)

We first tried a zero shot model, which makes a guess without ever training
on our data. It almost always guessed serious, and caught only 9 percent of
the true not serious reports. We then trained a small model on our own
labeled data, with extra weight given to the smaller not serious group so
the model could not ignore it. This raised the not serious catch rate to 71
percent, while overall accuracy rose from 80 to 88 percent. Both models were
tested on the exact same 1200 reports, which neither model trained on, so
the comparison is fair.

A real limit here: the reaction text is a short standard term, like Nausea,
not a full sentence written by a doctor. This is a fair task given the data
we have, but it is not deep reading of medical notes.

See `MODEL_COMPARISON.md` for the full write up of this part.

## Limitations

- FAERS reports are voluntary. Serious events are reported far more often
  than mild ones. This likely explains the high serious rate in this
  dataset.
- Almost every report lists many drugs, so a reaction can rarely be tied to
  one drug alone.
- All reports fall in one short time window, October 2013 to June 2014,
  since these were the first records the API returned. Results describe
  this period, not every year of FAERS data.
- These are patterns, not proof of cause. A drug may look different in the
  data because of the people who take it, not because of the drug itself.

## Project Structure

- `data_fetch.py`, gets raw data from the openFDA API
- `data_exploration.ipynb`, cleans the data and answers the six questions
- `zero_shot_baseline.py`, runs the zero shot model
- `zero_shot_visualize.py`, makes extra charts from the zero shot result
- `fine_tuning_model.py`, trains and tests the fine tuned model
- `compare_models.py`, makes the final comparison chart
- `results/`, all saved charts
- `MODEL_COMPARISON.md`, full write up of the model part

## Tech Stack

Python, pandas, requests, matplotlib, transformers, torch, scikit learn
