# Manual Ground Truth Evaluation

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
- Mean: 15.3%
- Median: 7.0%
- Deals: 10
- Discovery mode: suggest-sic+embedding (ladder: single-sic baseline -> suggest-sic expansion; compare runs by mode, the delta is the measured value of each discovery upgrade)
- Mean reachable (ranking-layer) precision: 78.6%

## Coverage Waterfall (all eligible banker comps)

| Stage | Count |
| --- | ---: |
| hit | 7 |
| ranked_but_not_top_k | 2 |
| low_confidence_filtered | 8 |
| no_valid_ev_ebitda | 2 |
| financial_filtered | 10 |
| truncated_by_embedding_candidate_limit | 12 |
| below_similarity_threshold | 1 |
| outside_candidate_set_top_n | 2 |
| outside_expanded_taxonomy | 27 |

9 of 71 eligible comps reached the scored pool; stages above the pair hit/ranked_but_not_top_k are ranking outcomes, everything below is a coverage loss (discovery, filters, or data gaps). `missing_market_cap` usually means an FMP quota/coverage miss — re-run after the quota resets before reading it as a data gap.

## Deal Detail
### manitex-2024 — MNTX (Manitex International, Inc.)
- Source: Brown Gibbons Lang & Company, 2024-11-12 — https://www.sec.gov/Archives/edgar/data/1302028/000119312524262120/d887136ddefm14a.htm
- Precision@15: 0.0% (reachable: 0.0%)
- Hits: none
- Missed before universe/scoring: TEX, HRI, URI
- Missed after ranking: MTW
- Excluded delisted banker comps: HEES
- Excluded non-US-filer banker comps: CGCBV.HE, PAL.VI, 6395.T, AHT.L
- Scored pool: 29 companies (21 selectable)
- Discovery SIC codes: primary 3559, 3537, 3531; adjacent 3713, 3714, 3569
- Loss stages: HRI=outside_expanded_taxonomy, MTW=ranked_but_not_top_k, TEX=financial_filtered, URI=outside_expanded_taxonomy

### pgt-innovations-2024 — PGTI (PGT Innovations, Inc.)
- Source: Evercore Group L.L.C., 2024-02-14 — https://www.sec.gov/Archives/edgar/data/1354327/000119312524036596/d728177ddefm14a.htm
- Precision@15: 0.0% (no comp reached ranking)
- Hits: none
- Missed before universe/scoring: APOG, FBIN, JHX, JELD, NX, TGLS, TREX
- Missed after ranking: GFF
- Excluded delisted banker comps: DOOR, AZEK
- Scored pool: 5 companies (3 selectable) — selection trivial (pool <= K, precision measures coverage only)
- Discovery SIC codes: primary 3442; adjacent 5211, 5031
- Loss stages: APOG=outside_expanded_taxonomy, FBIN=outside_expanded_taxonomy, GFF=low_confidence_filtered, JELD=outside_expanded_taxonomy, JHX=outside_expanded_taxonomy, NX=outside_expanded_taxonomy, TGLS=outside_expanded_taxonomy, TREX=outside_expanded_taxonomy

### circor-international-2023 — CIK1091883 (CIRCOR International, Inc.)
- Source: Evercore Group L.L.C. / J.P. Morgan Securities LLC (identical selected-companies sets), 2023-07-17 — https://www.sec.gov/Archives/edgar/data/1091883/000114036123034666/ny20009611x2_defm14a.htm
- Precision@15: 14.3% (reachable: 50.0%)
- Hits: GRC
- Missed before universe/scoring: WWD, CW, MOG-A
- Missed after ranking: ITT, FLS, CR
- Excluded delisted banker comps: TGI
- Excluded non-US-filer banker comps: SMIN.L, IMI.L, ROR.L
- Scored pool: 42 companies (29 selectable)
- Discovery SIC codes: primary 3490, 3561, 3728; adjacent 3829, 5084, 3714, 3531, 3443, 3724
- Loss stages: CR=low_confidence_filtered, CW=truncated_by_embedding_candidate_limit, FLS=ranked_but_not_top_k, ITT=low_confidence_filtered, MOG-A=truncated_by_embedding_candidate_limit, WWD=outside_expanded_taxonomy

### kaman-2024 — KAMN (Kaman Corporation)
- Source: J.P. Morgan Securities LLC, 2024-03-08 — https://www.sec.gov/Archives/edgar/data/54381/000114036124012403/ny20021849x2_defm14a.htm
- Precision@15: 0.0% (no comp reached ranking)
- Hits: none
- Missed before universe/scoring: TDG, RRX, RBC, HWM, HXL, AIN, DCO
- Missed after ranking: HEI
- Excluded delisted banker comps: B, SPR, TGI
- Scored pool: 24 companies (19 selectable)
- Discovery SIC codes: primary 3728; adjacent 3812, 3724, 3714, 3721
- Loss stages: AIN=outside_expanded_taxonomy, DCO=financial_filtered, HEI=low_confidence_filtered, HWM=outside_expanded_taxonomy, HXL=outside_expanded_taxonomy, RBC=outside_expanded_taxonomy, RRX=outside_expanded_taxonomy, TDG=financial_filtered

### barnes-group-2024 — B (Barnes Group Inc.)
- Source: Jefferies LLC (selected public companies analyses); Goldman Sachs & Co. LLC (separate opinion, no public-comps analysis), 2024-12-06 — https://www.sec.gov/Archives/edgar/data/9984/000114036124048581/ny20037086x2_defm14a.htm
- Precision@15: 7.7% (reachable: 100.0%)
- Hits: ATRO
- Missed before universe/scoring: AIN, ACA, BRC, CW, DCO, ESE, MOG-A, RBC, SPXC, TRS, WWD
- Missed after ranking: CR
- Excluded delisted banker comps: HI, TGI
- Scored pool: 33 companies (22 selectable)
- Discovery SIC codes: primary 3490, 3728, 3724, 3569; adjacent 3714, 3812, 3829, 3537
- Loss stages: ACA=truncated_by_embedding_candidate_limit, AIN=outside_expanded_taxonomy, BRC=outside_expanded_taxonomy, CR=low_confidence_filtered, CW=truncated_by_embedding_candidate_limit, DCO=financial_filtered, ESE=outside_expanded_taxonomy, MOG-A=truncated_by_embedding_candidate_limit, RBC=truncated_by_embedding_candidate_limit, SPXC=truncated_by_embedding_candidate_limit, TRS=truncated_by_embedding_candidate_limit, WWD=outside_expanded_taxonomy

### haynes-international-2024 — HAYN (Haynes International, Inc.)
- Source: Jefferies LLC, 2024-03-18 — https://www.sec.gov/Archives/edgar/data/858655/000110465924035231/tm248086-1_defm14a.htm
- Precision@15: 66.7% (reachable: 100.0%)
- Hits: ATI, CRS
- Missed before universe/scoring: HWM
- Missed after ranking: none
- Excluded delisted banker comps: USAP
- Excluded non-US-filer banker comps: ACX.MC, APAM.AS
- Scored pool: 15 companies (11 selectable) — selection trivial (pool <= K, precision measures coverage only)
- Discovery SIC codes: primary 3310; adjacent 3357, 3728, 3334, 3312
- Loss stages: HWM=financial_filtered

### universal-stainless-2024 — USAP (Universal Stainless & Alloy Products, Inc.)
- Source: TD Cowen (TD Securities (USA) LLC), 2024-11-27 — https://www.sec.gov/Archives/edgar/data/931584/000119312524267249/d842335ddefm14a.htm
- Precision@15: 33.3% (reachable: 100.0%)
- Hits: CRS
- Missed before universe/scoring: ATI, MTUS
- Missed after ranking: none
- Excluded delisted banker comps: none
- Excluded non-US-filer banker comps: ACX.MC, APAM.AS
- Scored pool: 4 companies (3 selectable) — selection trivial (pool <= K, precision measures coverage only)
- Discovery SIC codes: primary 3312; adjacent 3317
- Loss stages: ATI=financial_filtered, MTUS=no_valid_ev_ebitda

### l-s-starrett-2024 — SCX (The L.S. Starrett Company)
- Source: Lincoln International LLC, 2024-04-12 — https://www.sec.gov/Archives/edgar/data/93676/000143774924011842/scx20240411_defm14a.htm
- Precision@15: 25.0% (reachable: 100.0%)
- Hits: KMT
- Missed before universe/scoring: SWK, SNA, WOR
- Missed after ranking: none
- Excluded delisted banker comps: none
- Excluded non-US-filer banker comps: 6971.T, 002444.SZ, 6136.T
- Scored pool: 16 companies (12 selectable) — selection trivial (pool <= K, precision measures coverage only)
- Discovery SIC codes: primary 3420, 3823, 3829; adjacent 3541, 3569
- Loss stages: SNA=financial_filtered, SWK=financial_filtered, WOR=outside_expanded_taxonomy

### masonite-2024 — DOOR (Masonite International Corporation)
- Source: Goldman Sachs & Co. LLC and Jefferies LLC (separate opinions; ground truth is the union of their selected-companies lists), 2024-03-22 — https://www.sec.gov/Archives/edgar/data/893691/000119312524075345/d771808ddefm14a.htm
- Precision@15: 0.0% (no comp reached ranking)
- Hits: none
- Missed before universe/scoring: JELD, MBC, OC
- Missed after ranking: FBIN, GFF
- Excluded delisted banker comps: AMWD
- Scored pool: 15 companies (8 selectable) — selection trivial (pool <= K, precision measures coverage only)
- Discovery SIC codes: primary 2430, 3442; adjacent 5211, 5031
- Loss stages: FBIN=low_confidence_filtered, GFF=low_confidence_filtered, JELD=financial_filtered, MBC=outside_expanded_taxonomy, OC=outside_expanded_taxonomy

### chase-corp-2023 — CCF (Chase Corporation)
- Source: Perella Weinberg Partners LP, 2023-08-31 — https://www.sec.gov/Archives/edgar/data/830524/000114036123041904/ny20009924x2_defm14a.htm
- Precision@15: 6.2% (reachable: 100.0%)
- Hits: CSW
- Missed before universe/scoring: AVD, HWKN, NGVT, IOSP, MATV, SCL, UFPT, ASH, BCPC, CBT, ESI, FUL, KWR, ROG
- Missed after ranking: AVNT
- Excluded delisted banker comps: none
- Scored pool: 18 companies (14 selectable) — selection trivial (pool <= K, precision measures coverage only)
- Discovery SIC codes: primary 2891; adjacent 3081, 2821, 2851, 3714
- Loss stages: ASH=outside_expanded_taxonomy, AVD=truncated_by_embedding_candidate_limit, AVNT=low_confidence_filtered, BCPC=outside_candidate_set_top_n, CBT=truncated_by_embedding_candidate_limit, ESI=truncated_by_embedding_candidate_limit, FUL=no_valid_ev_ebitda, HWKN=outside_expanded_taxonomy, IOSP=below_similarity_threshold, KWR=outside_expanded_taxonomy, MATV=outside_expanded_taxonomy, NGVT=outside_candidate_set_top_n, ROG=financial_filtered, SCL=truncated_by_embedding_candidate_limit, UFPT=outside_expanded_taxonomy

