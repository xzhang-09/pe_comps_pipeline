# Manual Ground Truth Evaluation — suggest-sic rung

This file holds the **suggest-sic** discovery-ladder rung; the baseline
**single-sic** rung is in [`results.md`](results.md) — the delta between the
two is the measured value of the LLM-suggested-SIC discovery upgrade.

## Methodology

- Ground truth: fairness-opinion Selected Companies Analysis lists, audited against the filings
  (verbatim comp names, advisor, projected financials) — see eval/ground_truth/manual_deals.json
  per-deal notes for provenance.
- Denominator: only banker comps that are still public **and** US 10-K filers (us_filer=true)
  count toward Precision. Delisted comps and foreign-listed non-filers are excluded from the
  denominator but retained in the audit trail — the pipeline's documented data contract is US
  EDGAR 10-K filers, so those names are out of reach by design, not selection misses.
- Target financials: the projection year is the filing-date fiscal year when at least ~6 months
  of it remained, otherwise the next fiscal year; both years' figures are recorded in each
  deal's `source` field.
- Discovery vs. ranking: every eligible comp is attributed to a waterfall stage (below).
  `Reachable precision` scores the ranking layer alone — hits over comps that actually reached
  the scored pool — since a comp lost at discovery says nothing about ranking quality.

## Precision@15
- Mean: 19.3%
- Median: 17.7%
- Deals: 10
- Discovery mode: suggest-sic (ladder: single-sic baseline -> suggest-sic expansion; compare runs by mode, the delta is the measured value of each discovery upgrade)
- Mean reachable (ranking-layer) precision: 85.0%

## Coverage Waterfall (all eligible banker comps)

| Stage | Count |
| --- | ---: |
| hit | 12 |
| ranked_but_not_top_k | 3 |
| low_confidence_filtered | 2 |
| no_valid_ev_ebitda | 2 |
| financial_filtered | 9 |
| not_in_sic_universe | 43 |

15 of 71 eligible comps reached the scored pool; stages above the pair hit/ranked_but_not_top_k are ranking outcomes, everything below is a coverage loss (discovery, filters, or data gaps). `missing_market_cap` usually means an FMP quota/coverage miss — re-run after the quota resets before reading it as a data gap.

## Deal Detail
### manitex-2024 — MNTX (Manitex International, Inc.)
- Source: Brown Gibbons Lang & Company, 2024-11-12 — https://www.sec.gov/Archives/edgar/data/1302028/000119312524262120/d887136ddefm14a.htm
- Precision@15: 0.0% (reachable: 0.0%)
- Hits: none
- Missed before universe/scoring: TEX, HRI, URI
- Missed after ranking: MTW
- Excluded delisted banker comps: HEES
- Excluded non-US-filer banker comps: CGCBV.HE, PAL.VI, 6395.T, AHT.L
- Scored pool: 26 companies (24 selectable)
- Discovery SIC codes: primary 3559, 3537, 3531; adjacent 3713, 3714, 3569
- Loss stages: HRI=not_in_sic_universe, MTW=ranked_but_not_top_k, TEX=financial_filtered, URI=not_in_sic_universe

### pgt-innovations-2024 — PGTI (PGT Innovations, Inc.)
- Source: Evercore Group L.L.C., 2024-02-14 — https://www.sec.gov/Archives/edgar/data/1354327/000119312524036596/d728177ddefm14a.htm
- Precision@15: 12.5% (reachable: 100.0%)
- Hits: APOG
- Missed before universe/scoring: FBIN, JHX, JELD, NX, TGLS, TREX
- Missed after ranking: GFF
- Excluded delisted banker comps: DOOR, AZEK
- Scored pool: 5 companies (4 selectable) — selection trivial (pool <= K, precision measures coverage only)
- Discovery SIC codes: primary 3442, 3231; adjacent 5211, 5031, 3444
- Loss stages: FBIN=not_in_sic_universe, GFF=low_confidence_filtered, JELD=not_in_sic_universe, JHX=not_in_sic_universe, NX=not_in_sic_universe, TGLS=not_in_sic_universe, TREX=not_in_sic_universe

### circor-international-2023 — CIK1091883 (CIRCOR International, Inc.)
- Source: Evercore Group L.L.C. / J.P. Morgan Securities LLC (identical selected-companies sets), 2023-07-17 — https://www.sec.gov/Archives/edgar/data/1091883/000114036123034666/ny20009611x2_defm14a.htm
- Precision@15: 28.6% (reachable: 50.0%)
- Hits: CR, GRC
- Missed before universe/scoring: WWD, CW, MOG-A
- Missed after ranking: ITT, FLS
- Excluded delisted banker comps: TGI
- Excluded non-US-filer banker comps: SMIN.L, IMI.L, ROR.L
- Scored pool: 39 companies (39 selectable)
- Discovery SIC codes: primary 3490, 3561; adjacent 3728, 3822, 5084, 3714, 3569, 3724, 3443
- Loss stages: CW=not_in_sic_universe, FLS=ranked_but_not_top_k, ITT=ranked_but_not_top_k, MOG-A=not_in_sic_universe, WWD=not_in_sic_universe

### kaman-2024 — KAMN (Kaman Corporation)
- Source: J.P. Morgan Securities LLC, 2024-03-08 — https://www.sec.gov/Archives/edgar/data/54381/000114036124012403/ny20021849x2_defm14a.htm
- Precision@15: 12.5% (reachable: 100.0%)
- Hits: HEI
- Missed before universe/scoring: TDG, RRX, RBC, HWM, HXL, AIN, DCO
- Missed after ranking: none
- Excluded delisted banker comps: B, SPR, TGI
- Scored pool: 8 companies (8 selectable) — selection trivial (pool <= K, precision measures coverage only)
- Discovery SIC codes: primary 3728; adjacent 3724, 3721, 3452, 3812
- Loss stages: AIN=not_in_sic_universe, DCO=financial_filtered, HWM=not_in_sic_universe, HXL=not_in_sic_universe, RBC=not_in_sic_universe, RRX=not_in_sic_universe, TDG=financial_filtered

### barnes-group-2024 — B (Barnes Group Inc.)
- Source: Jefferies LLC (selected public companies analyses); Goldman Sachs & Co. LLC (separate opinion, no public-comps analysis), 2024-12-06 — https://www.sec.gov/Archives/edgar/data/9984/000114036124048581/ny20037086x2_defm14a.htm
- Precision@15: 15.4% (reachable: 100.0%)
- Hits: ATRO, CR
- Missed before universe/scoring: AIN, ACA, BRC, CW, DCO, ESE, MOG-A, RBC, SPXC, TRS, WWD
- Missed after ranking: none
- Excluded delisted banker comps: HI, TGI
- Scored pool: 34 companies (34 selectable)
- Discovery SIC codes: primary 3490, 3728, 3724, 3569; adjacent 3714, 3812, 3829, 3537
- Loss stages: ACA=not_in_sic_universe, AIN=not_in_sic_universe, BRC=not_in_sic_universe, CW=not_in_sic_universe, DCO=financial_filtered, ESE=not_in_sic_universe, MOG-A=not_in_sic_universe, RBC=not_in_sic_universe, SPXC=not_in_sic_universe, TRS=not_in_sic_universe, WWD=not_in_sic_universe

### haynes-international-2024 — HAYN (Haynes International, Inc.)
- Source: Jefferies LLC, 2024-03-18 — https://www.sec.gov/Archives/edgar/data/858655/000110465924035231/tm248086-1_defm14a.htm
- Precision@15: 33.3% (reachable: 100.0%)
- Hits: CRS
- Missed before universe/scoring: ATI, HWM
- Missed after ranking: none
- Excluded delisted banker comps: USAP
- Excluded non-US-filer banker comps: ACX.MC, APAM.AS
- Scored pool: 10 companies (10 selectable) — selection trivial (pool <= K, precision measures coverage only)
- Discovery SIC codes: primary 3310; adjacent 3334, 3357, 3728, 3312
- Loss stages: ATI=not_in_sic_universe, HWM=not_in_sic_universe

### universal-stainless-2024 — USAP (Universal Stainless & Alloy Products, Inc.)
- Source: TD Cowen (TD Securities (USA) LLC), 2024-11-27 — https://www.sec.gov/Archives/edgar/data/931584/000119312524267249/d842335ddefm14a.htm
- Precision@15: 33.3% (reachable: 100.0%)
- Hits: CRS
- Missed before universe/scoring: ATI, MTUS
- Missed after ranking: none
- Excluded delisted banker comps: none
- Excluded non-US-filer banker comps: ACX.MC, APAM.AS
- Scored pool: 3 companies (3 selectable) — selection trivial (pool <= K, precision measures coverage only)
- Discovery SIC codes: primary 3312; adjacent 3317, 3334
- Loss stages: ATI=financial_filtered, MTUS=no_valid_ev_ebitda

### l-s-starrett-2024 — SCX (The L.S. Starrett Company)
- Source: Lincoln International LLC, 2024-04-12 — https://www.sec.gov/Archives/edgar/data/93676/000143774924011842/scx20240411_defm14a.htm
- Precision@15: 25.0% (reachable: 100.0%)
- Hits: KMT
- Missed before universe/scoring: SWK, SNA, WOR
- Missed after ranking: none
- Excluded delisted banker comps: none
- Excluded non-US-filer banker comps: 6971.T, 002444.SZ, 6136.T
- Scored pool: 16 companies (16 selectable)
- Discovery SIC codes: primary 3420, 3823, 3829; adjacent 3541, 3569, 3825
- Loss stages: SNA=financial_filtered, SWK=financial_filtered, WOR=not_in_sic_universe

### masonite-2024 — DOOR (Masonite International Corporation)
- Source: Goldman Sachs & Co. LLC and Jefferies LLC (separate opinions; ground truth is the union of their selected-companies lists), 2024-03-22 — https://www.sec.gov/Archives/edgar/data/893691/000119312524075345/d771808ddefm14a.htm
- Precision@15: 20.0% (reachable: 100.0%)
- Hits: FBIN
- Missed before universe/scoring: JELD, MBC, OC
- Missed after ranking: GFF
- Excluded delisted banker comps: AMWD
- Scored pool: 5 companies (4 selectable) — selection trivial (pool <= K, precision measures coverage only)
- Discovery SIC codes: primary 2430, 3442; adjacent 5211, 5031
- Loss stages: GFF=low_confidence_filtered, JELD=financial_filtered, MBC=not_in_sic_universe, OC=not_in_sic_universe

### chase-corp-2023 — CCF (Chase Corporation)
- Source: Perella Weinberg Partners LP, 2023-08-31 — https://www.sec.gov/Archives/edgar/data/830524/000114036123041904/ny20009924x2_defm14a.htm
- Precision@15: 12.5% (reachable: 100.0%)
- Hits: CSW, AVNT
- Missed before universe/scoring: AVD, HWKN, NGVT, IOSP, MATV, SCL, UFPT, ASH, BCPC, CBT, ESI, FUL, KWR, ROG
- Missed after ranking: none
- Excluded delisted banker comps: none
- Scored pool: 6 companies (6 selectable) — selection trivial (pool <= K, precision measures coverage only)
- Discovery SIC codes: primary 2891; adjacent 3081, 2821, 2851, 3312, 5065
- Loss stages: ASH=not_in_sic_universe, AVD=not_in_sic_universe, BCPC=not_in_sic_universe, CBT=not_in_sic_universe, ESI=not_in_sic_universe, FUL=no_valid_ev_ebitda, HWKN=not_in_sic_universe, IOSP=not_in_sic_universe, KWR=not_in_sic_universe, MATV=not_in_sic_universe, NGVT=not_in_sic_universe, ROG=financial_filtered, SCL=not_in_sic_universe, UFPT=not_in_sic_universe

