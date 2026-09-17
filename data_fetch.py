import requests
import pandas as pd
import time
import os

BASE_URL = "https://api.fda.gov/drug/event.json"

DRUGS = ["metformin", "ibuprofen", "atorvastatin"]
RECORDS_PER_DRUG = 2000   
PAGE_SIZE = 100             
OUTPUT_PATH = "data/raw_adverse_events.csv"


def fetch_drug_events(drug_name: str, total_records: int, page_size: int) -> list[dict]:
    """Pull `total_records` adverse event reports for one drug, paginated."""
    all_results = []
    skip = 0

    while skip < total_records:
        limit = min(page_size, total_records - skip)
        params = {
            "search": f'patient.drug.medicinalproduct:"{drug_name}"',
            "limit": limit,
            "skip": skip,
        }
        response = requests.get(BASE_URL, params=params, timeout=30)

        if response.status_code == 404:
            # openFDA returns 404 when skip exceeds available results
            print(f"  No more results for {drug_name} at skip={skip}")
            break
        response.raise_for_status()

        batch = response.json().get("results", [])
        if not batch:
            break

        all_results.extend(batch)
        skip += page_size
        print(f"  {drug_name}: pulled {len(all_results)} records so far")

        time.sleep(0.3) 

    return all_results


def flatten_record(record: dict, queried_drug: str) -> list[dict]:
    """
    Turn one nested API record into one row per (drug, reaction) pair.
    Deliberately drops the bulky `openfda` enrichment block. it's not
    needed for this analysis and can balloon record size significantly.
    """
    rows = []
    patient = record.get("patient", {})
    drugs = patient.get("drug", [])
    reactions = patient.get("reaction", [])

    drug_names = [d.get("medicinalproduct") for d in drugs if d.get("medicinalproduct")]
    drug_routes = [d.get("drugadministrationroute") for d in drugs if d.get("drugadministrationroute")]

    for reaction in reactions:
        reaction_term = reaction.get("reactionmeddrapt")
        if not reaction_term:
            continue
        rows.append({
            "safetyreportid": record.get("safetyreportid"),
            "queried_drug": queried_drug,
            "drug_names_on_report": "; ".join(drug_names),
            "drug_routes_on_report": "; ".join(drug_routes) if drug_routes else None,
            "reaction_term": reaction_term,
            "reaction_outcome": reaction.get("reactionoutcome"),
            "serious": record.get("serious"),
            "seriousnessdeath": record.get("seriousnessdeath", "0"),
            "seriousnesshospitalization": record.get("seriousnesshospitalization", "0"),
            "seriousnesslifethreatening": record.get("seriousnesslifethreatening", "0"),
            "seriousnessdisabling": record.get("seriousnessdisabling", "0"),
            "seriousnesscongenitalanomali": record.get("seriousnesscongenitalanomali", "0"),
            "occurcountry": record.get("occurcountry"),
            "patient_sex": patient.get("patientsex"),
            "patient_age": patient.get("patientonsetage"),
            "receivedate": record.get("receivedate"),
        })
    return rows


def main():
    os.makedirs("data", exist_ok=True)
    all_rows = []

    for drug in DRUGS:
        print(f"Fetching {drug}...")
        raw_records = fetch_drug_events(drug, RECORDS_PER_DRUG, PAGE_SIZE)
        for record in raw_records:
            all_rows.extend(flatten_record(record, drug))

    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(df)} rows ({df['safetyreportid'].nunique()} unique reports) to {OUTPUT_PATH}")
    print("\nSample:")
    print(df.head())


if __name__ == "__main__":
    main()