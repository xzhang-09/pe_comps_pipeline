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
- Mean: 9.3%
- Median: 6.2%
- Deals: 16
- Discovery mode: single-sic (ladder: single-sic baseline -> suggest-sic expansion; compare runs by mode, the delta is the measured value of each discovery upgrade)
- Mean reachable (ranking-layer) precision: 92.6%

## Coverage Waterfall (all eligible banker comps)

| Stage | Count |
| --- | ---: |
| hit | 9 |
| ranked_but_not_top_k | 2 |
| no_valid_ev_ebitda | 3 |
| financial_filtered | 18 |
| not_in_sic_universe | 84 |

11 of 116 eligible comps reached the scored pool; stages above the pair hit/ranked_but_not_top_k are ranking outcomes, everything below is a coverage loss (discovery, filters, or data gaps). `missing_market_cap` usually means an FMP quota/coverage miss — re-run after the quota resets before reading it as a data gap.

## Deal Detail
### manitex-2024 — MNTX (Manitex International, Inc.)
- Source: Brown Gibbons Lang & Company, 2024-11-12 — https://www.sec.gov/Archives/edgar/data/1302028/000119312524262120/d887136ddefm14a.htm
- Precision@15: 0.0% (no comp reached ranking)
- Hits: none
- Missed before universe/scoring: MTW, TEX, HRI, URI
- Missed after ranking: none
- Excluded delisted banker comps: HEES
- Excluded non-US-filer banker comps: CGCBV.HE, PAL.VI, 6395.T, AHT.L
- Scored pool: 7 companies (7 selectable) — selection trivial (pool <= K, precision measures coverage only)
- Discovery SIC codes: primary 3559; adjacent none
- Loss stages: HRI=not_in_sic_universe, MTW=not_in_sic_universe, TEX=not_in_sic_universe, URI=not_in_sic_universe

### pgt-innovations-2024 — PGTI (PGT Innovations, Inc.)
- Source: Evercore Group L.L.C., 2024-02-14 — https://www.sec.gov/Archives/edgar/data/1354327/000119312524036596/d728177ddefm14a.htm
- Precision@15: 12.5% (reachable: 100.0%)
- Hits: GFF
- Missed before universe/scoring: APOG, FBIN, JHX, JELD, NX, TGLS, TREX
- Missed after ranking: none
- Excluded delisted banker comps: DOOR, AZEK
- Scored pool: 2 companies (2 selectable) — selection trivial (pool <= K, precision measures coverage only)
- Discovery SIC codes: primary 3442; adjacent none
- Loss stages: APOG=not_in_sic_universe, FBIN=not_in_sic_universe, JELD=not_in_sic_universe, JHX=not_in_sic_universe, NX=not_in_sic_universe, TGLS=not_in_sic_universe, TREX=not_in_sic_universe

### circor-international-2023 — CIK1091883 (CIRCOR International, Inc.)
- Source: Evercore Group L.L.C. / J.P. Morgan Securities LLC (identical selected-companies sets), 2023-07-17 — https://www.sec.gov/Archives/edgar/data/1091883/000114036123034666/ny20009611x2_defm14a.htm
- Precision@15: 14.3% (reachable: 100.0%)
- Hits: CR
- Missed before universe/scoring: WWD, CW, ITT, FLS, MOG-A, GRC
- Missed after ranking: none
- Excluded delisted banker comps: TGI
- Excluded non-US-filer banker comps: SMIN.L, IMI.L, ROR.L
- Scored pool: 5 companies (5 selectable) — selection trivial (pool <= K, precision measures coverage only)
- Discovery SIC codes: primary 3490; adjacent none
- Loss stages: CW=not_in_sic_universe, FLS=not_in_sic_universe, GRC=not_in_sic_universe, ITT=not_in_sic_universe, MOG-A=not_in_sic_universe, WWD=not_in_sic_universe

### kaman-2024 — KAMN (Kaman Corporation)
- Source: J.P. Morgan Securities LLC, 2024-03-08 — https://www.sec.gov/Archives/edgar/data/54381/000114036124012403/ny20021849x2_defm14a.htm
- Precision@15: 0.0% (no comp reached ranking)
- Hits: none
- Missed before universe/scoring: TDG, HEI, RRX, RBC, HWM, HXL, AIN, DCO
- Missed after ranking: none
- Excluded delisted banker comps: B, SPR, TGI
- Scored pool: 3 companies (3 selectable) — selection trivial (pool <= K, precision measures coverage only)
- Discovery SIC codes: primary 3728; adjacent none
- Loss stages: AIN=not_in_sic_universe, DCO=financial_filtered, HEI=not_in_sic_universe, HWM=not_in_sic_universe, HXL=not_in_sic_universe, RBC=not_in_sic_universe, RRX=not_in_sic_universe, TDG=financial_filtered

### barnes-group-2024 — B (Barnes Group Inc.)
- Source: Jefferies LLC (selected public companies analyses); Goldman Sachs & Co. LLC (separate opinion, no public-comps analysis), 2024-12-06 — https://www.sec.gov/Archives/edgar/data/9984/000114036124048581/ny20037086x2_defm14a.htm
- Precision@15: 7.7% (reachable: 100.0%)
- Hits: CR
- Missed before universe/scoring: AIN, ACA, ATRO, BRC, CW, DCO, ESE, MOG-A, RBC, SPXC, TRS, WWD
- Missed after ranking: none
- Excluded delisted banker comps: HI, TGI
- Scored pool: 5 companies (5 selectable) — selection trivial (pool <= K, precision measures coverage only)
- Discovery SIC codes: primary 3490; adjacent none
- Loss stages: ACA=not_in_sic_universe, AIN=not_in_sic_universe, ATRO=not_in_sic_universe, BRC=not_in_sic_universe, CW=not_in_sic_universe, DCO=not_in_sic_universe, ESE=not_in_sic_universe, MOG-A=not_in_sic_universe, RBC=not_in_sic_universe, SPXC=not_in_sic_universe, TRS=not_in_sic_universe, WWD=not_in_sic_universe

### haynes-international-2024 — HAYN (Haynes International, Inc.)
- Source: Jefferies LLC, 2024-03-18 — https://www.sec.gov/Archives/edgar/data/858655/000110465924035231/tm248086-1_defm14a.htm
- Precision@15: 0.0% (no comp reached ranking)
- Hits: none
- Missed before universe/scoring: ATI, CRS, HWM
- Missed after ranking: none
- Excluded delisted banker comps: USAP
- Excluded non-US-filer banker comps: ACX.MC, APAM.AS
- Scored pool: 2 companies (2 selectable) — selection trivial (pool <= K, precision measures coverage only)
- Discovery SIC codes: primary 3310; adjacent none
- Loss stages: ATI=not_in_sic_universe, CRS=not_in_sic_universe, HWM=not_in_sic_universe

### universal-stainless-2024 — USAP (Universal Stainless & Alloy Products, Inc.)
- Source: TD Cowen (TD Securities (USA) LLC), 2024-11-27 — https://www.sec.gov/Archives/edgar/data/931584/000119312524267249/d842335ddefm14a.htm
- Precision@15: 33.3% (reachable: 100.0%)
- Hits: CRS
- Missed before universe/scoring: ATI, MTUS
- Missed after ranking: none
- Excluded delisted banker comps: none
- Excluded non-US-filer banker comps: ACX.MC, APAM.AS
- Scored pool: 2 companies (2 selectable) — selection trivial (pool <= K, precision measures coverage only)
- Discovery SIC codes: primary 3312; adjacent none
- Loss stages: ATI=not_in_sic_universe, MTUS=no_valid_ev_ebitda

### l-s-starrett-2024 — SCX (The L.S. Starrett Company)
- Source: Lincoln International LLC, 2024-04-12 — https://www.sec.gov/Archives/edgar/data/93676/000143774924011842/scx20240411_defm14a.htm
- Precision@15: 0.0% (no comp reached ranking)
- Hits: none
- Missed before universe/scoring: SWK, SNA, WOR, KMT
- Missed after ranking: none
- Excluded delisted banker comps: none
- Excluded non-US-filer banker comps: 6971.T, 002444.SZ, 6136.T
- Scored pool: 5 companies (5 selectable) — selection trivial (pool <= K, precision measures coverage only)
- Discovery SIC codes: primary 3420; adjacent none
- Loss stages: KMT=not_in_sic_universe, SNA=financial_filtered, SWK=financial_filtered, WOR=not_in_sic_universe

### masonite-2024 — DOOR (Masonite International Corporation)
- Source: Goldman Sachs & Co. LLC and Jefferies LLC (separate opinions; ground truth is the union of their selected-companies lists), 2024-03-22 — https://www.sec.gov/Archives/edgar/data/893691/000119312524075345/d771808ddefm14a.htm
- Precision@15: 20.0% (reachable: 100.0%)
- Hits: FBIN
- Missed before universe/scoring: GFF, JELD, MBC, OC
- Missed after ranking: none
- Excluded delisted banker comps: AMWD
- Scored pool: 2 companies (2 selectable) — selection trivial (pool <= K, precision measures coverage only)
- Discovery SIC codes: primary 2430; adjacent none
- Loss stages: GFF=not_in_sic_universe, JELD=financial_filtered, MBC=not_in_sic_universe, OC=not_in_sic_universe

### chase-corp-2023 — CCF (Chase Corporation)
- Source: Perella Weinberg Partners LP, 2023-08-31 — https://www.sec.gov/Archives/edgar/data/830524/000114036123041904/ny20009924x2_defm14a.htm
- Precision@15: 6.2% (reachable: 100.0%)
- Hits: CSW
- Missed before universe/scoring: AVD, HWKN, NGVT, IOSP, MATV, SCL, UFPT, ASH, AVNT, BCPC, CBT, ESI, FUL, KWR, ROG
- Missed after ranking: none
- Excluded delisted banker comps: none
- Scored pool: 1 companies (1 selectable) — selection trivial (pool <= K, precision measures coverage only)
- Discovery SIC codes: primary 2891; adjacent none
- Loss stages: ASH=not_in_sic_universe, AVD=not_in_sic_universe, AVNT=not_in_sic_universe, BCPC=not_in_sic_universe, CBT=not_in_sic_universe, ESI=not_in_sic_universe, FUL=no_valid_ev_ebitda, HWKN=not_in_sic_universe, IOSP=not_in_sic_universe, KWR=not_in_sic_universe, MATV=not_in_sic_universe, NGVT=not_in_sic_universe, ROG=not_in_sic_universe, SCL=not_in_sic_universe, UFPT=not_in_sic_universe

### perficient-inc-2024 — CIK1085869 (PERFICIENT INC)
- Source: Wells Fargo, 2024-06-26 — https://www.sec.gov/Archives/edgar/data/1085869/000119312524169200/d834503ddefm14a.htm
- Precision@15: 25.0% (reachable: 100.0%)
- Hits: EPAM
- Missed before universe/scoring: GD, ACN, CTSH
- Missed after ranking: none
- Excluded delisted banker comps: TWKS
- Excluded non-US-filer banker comps: GLOB, DAVA, TCS.NS, INFY, CAP.PA, WIT, GIB
- Scored pool: 4 companies (4 selectable) — selection trivial (pool <= K, precision measures coverage only)
- Discovery SIC codes: primary 7371; adjacent none
- Loss stages: ACN=not_in_sic_universe, CTSH=financial_filtered, GD=not_in_sic_universe

### squarespace-inc-2024 — SQSP (Squarespace, Inc.)
- Source: Centerview Partners LLC, 2024-08-22 — https://www.sec.gov/Archives/edgar/data/1496963/000114036124038138/ny20030653x9_defm14a.htm
- Precision@15: 0.0% (no comp reached ranking)
- Hits: none
- Missed before universe/scoring: BIGC, BILL, GDDY, INTU, VRSN
- Missed after ranking: none
- Excluded delisted banker comps: none
- Excluded non-US-filer banker comps: WIX
- Scored pool: 48 companies (47 selectable)
- Discovery SIC codes: primary 7372; adjacent none
- Loss stages: BIGC=not_in_sic_universe, BILL=financial_filtered, GDDY=not_in_sic_universe, INTU=financial_filtered, VRSN=not_in_sic_universe

### smartsheet-inc-2024 — CIK1366561 (SMARTSHEET INC)
- Source: Qatalyst Partners, 2024-11-04 — https://www.sec.gov/Archives/edgar/data/1366561/000162828024044846/smartsheetdefm14a1142024.htm
- Precision@15: 5.9% (reachable: 33.3%)
- Hits: PAYC
- Missed before universe/scoring: WDAY, PATH, BILL, WK, FRSH, ASAN, FIVN, PD, SPT, CRM, TEAM, ZM, DBX, BXCAP
- Missed after ranking: OKTA, DOCU
- Excluded delisted banker comps: none
- Excluded non-US-filer banker comps: MNDY
- Scored pool: 47 companies (46 selectable)
- Discovery SIC codes: primary 7372; adjacent none
- Loss stages: ASAN=financial_filtered, BILL=financial_filtered, BXCAP=not_in_sic_universe, CRM=financial_filtered, DBX=no_valid_ev_ebitda, DOCU=ranked_but_not_top_k, FIVN=not_in_sic_universe, FRSH=financial_filtered, OKTA=ranked_but_not_top_k, PATH=financial_filtered, PD=financial_filtered, SPT=financial_filtered, TEAM=financial_filtered, WDAY=not_in_sic_universe, WK=financial_filtered, ZM=not_in_sic_universe

### stericycle-inc-2024 — CIK861878 (STERICYCLE INC)
- Source: BofA Securities, 2024-07-10 — https://www.sec.gov/Archives/edgar/data/861878/000119312524177292/d768297ddefm14a.htm
- Precision@15: 14.3% (reachable: 100.0%)
- Hits: CLH
- Missed before universe/scoring: RSG, WCN, WM, ARCC, UNF, VSTS
- Missed after ranking: none
- Excluded delisted banker comps: none
- Excluded non-US-filer banker comps: GFL, RTO, BFSA.DE, ELSSF, RWI.L, RST.L, SCHP.PA, VIE.PA
- Scored pool: 1 companies (1 selectable) — selection trivial (pool <= K, precision measures coverage only)
- Discovery SIC codes: primary 4955; adjacent none
- Loss stages: ARCC=not_in_sic_universe, RSG=not_in_sic_universe, UNF=not_in_sic_universe, VSTS=not_in_sic_universe, WCN=not_in_sic_universe, WM=not_in_sic_universe

### duckhorn-portfolio-inc-2024 — CIK1835256 (Duckhorn Portfolio, Inc.)
- Source: J.P. Morgan, 2024-12-02 — https://www.sec.gov/Archives/edgar/data/1835256/000119312524268428/d852182ddefm14a.htm
- Precision@15: N/A (no comp reached ranking)
- Hits: none
- Missed before universe/scoring: none
- Missed after ranking: none
- Excluded delisted banker comps: none
- Excluded non-US-filer banker comps: TWE.AX, CXMSF, RCO.PA
- Scored pool: 1 companies (1 selectable) — selection trivial (pool <= K, precision measures coverage only)
- Discovery SIC codes: primary 2080; adjacent none

### r1-rcm-inc-de-2024 — CIK1910851 (R1 RCM Inc. /DE)
- Source: Qatalyst Partners, 2024-10-16 — https://www.sec.gov/Archives/edgar/data/1910851/000119312524238130/d837225ddefm14a.htm
- Precision@15: 0.0% (no comp reached ranking)
- Hits: none
- Missed before universe/scoring: PHR, HQY, WAY, HSTM, EVH, OMCL, HCAT, PGNY, TBRG, PREM, EXLS, G
- Missed after ranking: none
- Excluded delisted banker comps: none
- Excluded non-US-filer banker comps: NU
- Scored pool: 1 companies (1 selectable) — selection trivial (pool <= K, precision measures coverage only)
- Discovery SIC codes: primary 8741; adjacent none
- Loss stages: EVH=financial_filtered, EXLS=not_in_sic_universe, G=not_in_sic_universe, HCAT=not_in_sic_universe, HQY=not_in_sic_universe, HSTM=not_in_sic_universe, OMCL=not_in_sic_universe, PGNY=not_in_sic_universe, PHR=not_in_sic_universe, PREM=not_in_sic_universe, TBRG=not_in_sic_universe, WAY=not_in_sic_universe

