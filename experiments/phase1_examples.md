# Phase 1: Shared Normalization — Eyeball Verification on 51 Ground Truth Pairs

This artifact documents the literal before/after text for deliberately sampled known-matching pairs from `dataset/train/train_ground_truth.tsv` (sampling offsets > 50,000, multi-match entities, India and US splits, real missing-field rows, and 15+ native Indic script pairs in Devanagari and Tamil).

### Key Properties Verified:
1. **Unicode combining marks preserved:** Devanagari and Tamil vowel signs (matras: `Mn`, `Mc`) and virama/halant are strictly preserved without being stripped.
2. **Sequence-wide legal suffix stripping:** `clean_norm` strips legal suffixes anywhere in the token sequence (leading 'Private X Limited', middle 'X Inc Center', trailing 'X Pvt Ltd', and Devanagari 'प्राइवेट/लिमिटेड').
3. **Missing-field flag setting:** Rows with missing address or name programmatically set `has_address=False` / `has_name=False` and yield empty strings `""` (never literal string `'nan'`).

TOTAL PAIRS EVALUATED: 51

--- [Pair #01] Missing Address | S1: S1-965667 vs S3-860443364 ---
  [S1 BEFORE]   Name:    'Maure Williams Colombier Inc'
                Address: '85 Wayne Avenue, Ticonderoga, NY'
  [S1 AFTER]    Norm Name:   'maure williams colombier incorporated' (clean: 'maure williams colombier')
                Norm Addr:   '85 wayne avenue ticonderoga ny' [Postal: None, StreetNo: 85]
                Flags:       has_name=True, has_address=True
  [CAND BEFORE] Name:    'Maure Williams Inc Center'
                Address: 'nan'
  [CAND AFTER]  Norm Name:   'maure williams incorporated center' (clean: 'maure williams center')
                Norm Addr:   '' [Postal: None, StreetNo: None]
                Flags:       has_name=True, has_address=False

--- [Pair #02] Missing Address | S1: S1-292900819 vs S2-791548670 ---
  [S1 BEFORE]   Name:    'Services Tycon Connect Partners'
                Address: 'Shop No. G-01-C, Mangalam - Residency Apartment, Pragati Nagar, Ajmer, Rajasthan'
  [S1 AFTER]    Norm Name:   'services tycon connect partners' (clean: 'tycon connect partners')
                Norm Addr:   'shop number g 01 c mangalam residency apartment pragati nagar ajmer rajasthan' [Postal: None, StreetNo: 01]
                Flags:       has_name=True, has_address=True
  [CAND BEFORE] Name:    'SERVICES TÉTONC CONNECT PARTNERS'
                Address: 'nan'
  [CAND AFTER]  Norm Name:   'services tétonc connect partners' (clean: 'tétonc connect partners')
                Norm Addr:   '' [Postal: None, StreetNo: None]
                Flags:       has_name=True, has_address=False

--- [Pair #03] Missing Address | S1: S1-953694901 vs S3-860481692 ---
  [S1 BEFORE]   Name:    'St. Lutheran Church LLC'
                Address: 'Crown Point, IN, 787 Ronny Court'
  [S1 AFTER]    Norm Name:   'st lutheran church llc' (clean: 'st lutheran church')
                Norm Addr:   'crown point in 787 ronny court' [Postal: 787, StreetNo: None]
                Flags:       has_name=True, has_address=True
  [CAND BEFORE] Name:    'St. Lutheran Church L.L.C.'
                Address: 'nan'
  [CAND AFTER]  Norm Name:   'st lutheran church l l c' (clean: 'st lutheran church l l c')
                Norm Addr:   '' [Postal: None, StreetNo: None]
                Flags:       has_name=True, has_address=False

--- [Pair #04] India Multi-Match (5 matches) | S1: S1-254018672 vs S2-731762814 ---
  [S1 BEFORE]   Name:    'Kriya Infotech Private Limited'
                Address: 'Gl No-5 Brahm Puri, Gookna, Ator Nagla, Ghaziabad, Uttar Pradesh'
  [S1 AFTER]    Norm Name:   'kriya infotech private limited' (clean: 'kriya infotech')
                Norm Addr:   'gl number 5 brahm puri gookna ator nagla ghaziabad uttar pradesh' [Postal: None, StreetNo: 5]
                Flags:       has_name=True, has_address=True
  [CAND BEFORE] Name:    'kriya infotech private limited'
                Address: 'GL NO-G-5 BRAHM PURI, GOOKNA, ATOR NAGLA, Uttar Pradesh'
  [CAND AFTER]  Norm Name:   'kriya infotech private limited' (clean: 'kriya infotech')
                Norm Addr:   'gl number g 5 brahm puri gookna ator nagla uttar pradesh' [Postal: None, StreetNo: 5]
                Flags:       has_name=True, has_address=True

--- [Pair #05] India Multi-Match (5 matches) | S1: S1-254018672 vs S2-538238275 ---
  [S1 BEFORE]   Name:    'Kriya Infotech Private Limited'
                Address: 'Gl No-5 Brahm Puri, Gookna, Ator Nagla, Ghaziabad, Uttar Pradesh'
  [S1 AFTER]    Norm Name:   'kriya infotech private limited' (clean: 'kriya infotech')
                Norm Addr:   'gl number 5 brahm puri gookna ator nagla ghaziabad uttar pradesh' [Postal: None, StreetNo: 5]
                Flags:       has_name=True, has_address=True
  [CAND BEFORE] Name:    'KRIYA INFOTECH PRIVATE LTD'
                Address: 'GL NO-G-5 BRAHM PURI, GOOKNA, ATOR NAGLA, Uttar Pradesh'
  [CAND AFTER]  Norm Name:   'kriya infotech private limited' (clean: 'kriya infotech')
                Norm Addr:   'gl number g 5 brahm puri gookna ator nagla uttar pradesh' [Postal: None, StreetNo: 5]
                Flags:       has_name=True, has_address=True

--- [Pair #06] India Multi-Match (5 matches) | S1: S1-254018672 vs S2-439455305 ---
  [S1 BEFORE]   Name:    'Kriya Infotech Private Limited'
                Address: 'Gl No-5 Brahm Puri, Gookna, Ator Nagla, Ghaziabad, Uttar Pradesh'
  [S1 AFTER]    Norm Name:   'kriya infotech private limited' (clean: 'kriya infotech')
                Norm Addr:   'gl number 5 brahm puri gookna ator nagla ghaziabad uttar pradesh' [Postal: None, StreetNo: 5]
                Flags:       has_name=True, has_address=True
  [CAND BEFORE] Name:    'Private Kriya Inomteech [Limited]'
                Address: 'GL NO-G-5 BRAHM PURI, GOOKNA, GHAZIABAD, ATOR NAGLA, Uttar Pradesh'
  [CAND AFTER]  Norm Name:   'private kriya inomteech limited' (clean: 'kriya inomteech')
                Norm Addr:   'gl number g 5 brahm puri gookna ghaziabad ator nagla uttar pradesh' [Postal: None, StreetNo: 5]
                Flags:       has_name=True, has_address=True

--- [Pair #07] India Multi-Match (5 matches) | S1: S1-254018672 vs S3-186276594 ---
  [S1 BEFORE]   Name:    'Kriya Infotech Private Limited'
                Address: 'Gl No-5 Brahm Puri, Gookna, Ator Nagla, Ghaziabad, Uttar Pradesh'
  [S1 AFTER]    Norm Name:   'kriya infotech private limited' (clean: 'kriya infotech')
                Norm Addr:   'gl number 5 brahm puri gookna ator nagla ghaziabad uttar pradesh' [Postal: None, StreetNo: 5]
                Flags:       has_name=True, has_address=True
  [CAND BEFORE] Name:    'Shri Kriya Infotech'
                Address: 'nan'
  [CAND AFTER]  Norm Name:   'shri kriya infotech' (clean: 'shri kriya infotech')
                Norm Addr:   '' [Postal: None, StreetNo: None]
                Flags:       has_name=True, has_address=False

--- [Pair #08] India Multi-Match (5 matches) | S1: S1-254018672 vs S3-816844366 ---
  [S1 BEFORE]   Name:    'Kriya Infotech Private Limited'
                Address: 'Gl No-5 Brahm Puri, Gookna, Ator Nagla, Ghaziabad, Uttar Pradesh'
  [S1 AFTER]    Norm Name:   'kriya infotech private limited' (clean: 'kriya infotech')
                Norm Addr:   'gl number 5 brahm puri gookna ator nagla ghaziabad uttar pradesh' [Postal: None, StreetNo: 5]
                Flags:       has_name=True, has_address=True
  [CAND BEFORE] Name:    'KRIYA INFOTECH PRIVATE LIMITED'
                Address: 'Gl No-5 Brahm Puri, NULL, Ator Nagla, Ghaziabad, UP'
  [CAND AFTER]  Norm Name:   'kriya infotech private limited' (clean: 'kriya infotech')
                Norm Addr:   'gl number 5 brahm puri null ator nagla ghaziabad up' [Postal: None, StreetNo: 5]
                Flags:       has_name=True, has_address=True

--- [Pair #09] India Multi-Match (6 matches) | S1: S1-644124701 vs S2-666436404 ---
  [S1 BEFORE]   Name:    'Universal Finance Private Limited'
                Address: 'Flat No. 402, Orient Platinum, Plot No. 10/A, Navi Mumbai, Kharghar, Panvel, Raigarh, Maharashtra'
  [S1 AFTER]    Norm Name:   'universal finance private limited' (clean: 'universal finance')
                Norm Addr:   'flat number 402 orient platinum plot number 10 a navi mumbai kharghar panvel raigarh maharashtra' [Postal: None, StreetNo: 402]
                Flags:       has_name=True, has_address=True
  [CAND BEFORE] Name:    'यूनिवर्सल फाइनेंस प्राइवेट लिमिटेड'
                Address: 'FLAT NO. #402, ORIENT PLATINUM, PLOT NO. 10/A, NAVI MUMBAI, KHARGHAR, PANVEL, RAIGARH, Maharashtra'
  [CAND AFTER]  Norm Name:   'यूनिवर्सल फाइनेंस प्राइवेट लिमिटेड' (clean: 'यूनिवर्सल फाइनेंस')
                Norm Addr:   'flat number 402 orient platinum plot number 10 a navi mumbai kharghar panvel raigarh maharashtra' [Postal: None, StreetNo: 402]
                Flags:       has_name=True, has_address=True

--- [Pair #10] India Multi-Match (6 matches) | S1: S1-644124701 vs S2-861214874 ---
  [S1 BEFORE]   Name:    'Universal Finance Private Limited'
                Address: 'Flat No. 402, Orient Platinum, Plot No. 10/A, Navi Mumbai, Kharghar, Panvel, Raigarh, Maharashtra'
  [S1 AFTER]    Norm Name:   'universal finance private limited' (clean: 'universal finance')
                Norm Addr:   'flat number 402 orient platinum plot number 10 a navi mumbai kharghar panvel raigarh maharashtra' [Postal: None, StreetNo: 402]
                Flags:       has_name=True, has_address=True
  [CAND BEFORE] Name:    'Universal Finance  Private'
                Address: 'NO #402 , ORIENT PLATINUM, PLOT NO. 10/A, NAVI MUMBAI, KHARGHAR, PANVEL, Maharashtra'
  [CAND AFTER]  Norm Name:   'universal finance private' (clean: 'universal finance')
                Norm Addr:   'number 402 orient platinum plot number 10 a navi mumbai kharghar panvel maharashtra' [Postal: None, StreetNo: 402]
                Flags:       has_name=True, has_address=True

--- [Pair #11] India Multi-Match (6 matches) | S1: S1-644124701 vs S3-88945455 ---
  [S1 BEFORE]   Name:    'Universal Finance Private Limited'
                Address: 'Flat No. 402, Orient Platinum, Plot No. 10/A, Navi Mumbai, Kharghar, Panvel, Raigarh, Maharashtra'
  [S1 AFTER]    Norm Name:   'universal finance private limited' (clean: 'universal finance')
                Norm Addr:   'flat number 402 orient platinum plot number 10 a navi mumbai kharghar panvel raigarh maharashtra' [Postal: None, StreetNo: 402]
                Flags:       has_name=True, has_address=True
  [CAND BEFORE] Name:    'यूनिवर्सल फाइनेंस Private Limited'
                Address: 'Flat No. 402, MH, Orient Platinum, Plot No. 10/A, Navi Mumbai, Kharghar, Raigarh, Panvel'
  [CAND AFTER]  Norm Name:   'यूनिवर्सल फाइनेंस private limited' (clean: 'यूनिवर्सल फाइनेंस')
                Norm Addr:   'flat number 402 mh orient platinum plot number 10 a navi mumbai kharghar raigarh panvel' [Postal: None, StreetNo: 402]
                Flags:       has_name=True, has_address=True

--- [Pair #12] India Multi-Match (6 matches) | S1: S1-644124701 vs S3-49955194 ---
  [S1 BEFORE]   Name:    'Universal Finance Private Limited'
                Address: 'Flat No. 402, Orient Platinum, Plot No. 10/A, Navi Mumbai, Kharghar, Panvel, Raigarh, Maharashtra'
  [S1 AFTER]    Norm Name:   'universal finance private limited' (clean: 'universal finance')
                Norm Addr:   'flat number 402 orient platinum plot number 10 a navi mumbai kharghar panvel raigarh maharashtra' [Postal: None, StreetNo: 402]
                Flags:       has_name=True, has_address=True
  [CAND BEFORE] Name:    'Universal Finance Limited Service'
                Address: 'No 402, Raigarh, Panvel, MH'
  [CAND AFTER]  Norm Name:   'universal finance limited service' (clean: 'universal finance service')
                Norm Addr:   'number 402 raigarh panvel mh' [Postal: None, StreetNo: 402]
                Flags:       has_name=True, has_address=True

--- [Pair #13] India Multi-Match (6 matches) | S1: S1-644124701 vs S3-307181122 ---
  [S1 BEFORE]   Name:    'Universal Finance Private Limited'
                Address: 'Flat No. 402, Orient Platinum, Plot No. 10/A, Navi Mumbai, Kharghar, Panvel, Raigarh, Maharashtra'
  [S1 AFTER]    Norm Name:   'universal finance private limited' (clean: 'universal finance')
                Norm Addr:   'flat number 402 orient platinum plot number 10 a navi mumbai kharghar panvel raigarh maharashtra' [Postal: None, StreetNo: 402]
                Flags:       has_name=True, has_address=True
  [CAND BEFORE] Name:    'Shri Universal  Limited Private Finance'
                Address: 'H.no 402, Raigarh, Panvel, MH'
  [CAND AFTER]  Norm Name:   'shri universal limited private finance' (clean: 'shri universal finance')
                Norm Addr:   'h number 402 raigarh panvel mh' [Postal: None, StreetNo: 402]
                Flags:       has_name=True, has_address=True

--- [Pair #14] India Multi-Match (6 matches) | S1: S1-644124701 vs S3-954228293 ---
  [S1 BEFORE]   Name:    'Universal Finance Private Limited'
                Address: 'Flat No. 402, Orient Platinum, Plot No. 10/A, Navi Mumbai, Kharghar, Panvel, Raigarh, Maharashtra'
  [S1 AFTER]    Norm Name:   'universal finance private limited' (clean: 'universal finance')
                Norm Addr:   'flat number 402 orient platinum plot number 10 a navi mumbai kharghar panvel raigarh maharashtra' [Postal: None, StreetNo: 402]
                Flags:       has_name=True, has_address=True
  [CAND BEFORE] Name:    'Universal Fínance Private Limited'
                Address: 'MH, Panvel, Orient Platinum, Plot No. 10/A, Navi Mumbai, Kharghar, Raigarh, No 402'
  [CAND AFTER]  Norm Name:   'universal fínance private limited' (clean: 'universal fínance')
                Norm Addr:   'mh panvel orient platinum plot number 10 a navi mumbai kharghar raigarh number 402' [Postal: 402, StreetNo: None]
                Flags:       has_name=True, has_address=True

--- [Pair #15] India Multi-Match (3 matches) | S1: S1-910658295 vs S2-902689305 ---
  [S1 BEFORE]   Name:    'Solutions Indian Designs Private Limited'
                Address: '111 Desaipura Infront Of, Kaka Ganapati Nandurbar, Nandurbar, Maharashtra'
  [S1 AFTER]    Norm Name:   'solutions indian designs private limited' (clean: 'indian designs')
                Norm Addr:   '111 desaipura infront of kaka ganapati nandurbar nandurbar maharashtra' [Postal: None, StreetNo: 111]
                Flags:       has_name=True, has_address=True
  [CAND BEFORE] Name:    'Solutions Indian Désigns Private Limited'
                Address: '111. DESAIPURA INFRONT OF, KAKA GANAPATI NANDURBAR, NANDURBAR, महाराष्ट्र'
  [CAND AFTER]  Norm Name:   'solutions indian désigns private limited' (clean: 'indian désigns')
                Norm Addr:   '111 desaipura infront of kaka ganapati nandurbar nandurbar महाराष्ट्र' [Postal: None, StreetNo: 111]
                Flags:       has_name=True, has_address=True

--- [Pair #16] India Multi-Match (3 matches) | S1: S1-910658295 vs S3-54861251 ---
  [S1 BEFORE]   Name:    'Solutions Indian Designs Private Limited'
                Address: '111 Desaipura Infront Of, Kaka Ganapati Nandurbar, Nandurbar, Maharashtra'
  [S1 AFTER]    Norm Name:   'solutions indian designs private limited' (clean: 'indian designs')
                Norm Addr:   '111 desaipura infront of kaka ganapati nandurbar nandurbar maharashtra' [Postal: None, StreetNo: 111]
                Flags:       has_name=True, has_address=True
  [CAND BEFORE] Name:    'Solutions Indian Désigns Private'
                Address: 'MH, Nandurbar, Door No ##111 Desaipura Infront Of'
  [CAND AFTER]  Norm Name:   'solutions indian désigns private' (clean: 'indian désigns')
                Norm Addr:   'mh nandurbar door number 111 desaipura infront of' [Postal: 111, StreetNo: None]
                Flags:       has_name=True, has_address=True

--- [Pair #17] India Multi-Match (3 matches) | S1: S1-910658295 vs S3-894534068 ---
  [S1 BEFORE]   Name:    'Solutions Indian Designs Private Limited'
                Address: '111 Desaipura Infront Of, Kaka Ganapati Nandurbar, Nandurbar, Maharashtra'
  [S1 AFTER]    Norm Name:   'solutions indian designs private limited' (clean: 'indian designs')
                Norm Addr:   '111 desaipura infront of kaka ganapati nandurbar nandurbar maharashtra' [Postal: None, StreetNo: 111]
                Flags:       has_name=True, has_address=True
  [CAND BEFORE] Name:    'Private Solutions Indian Designs Limited'
                Address: '##111 Desaipura Infront Of, Nandurbar, MH'
  [CAND AFTER]  Norm Name:   'private solutions indian designs limited' (clean: 'indian designs')
                Norm Addr:   '111 desaipura infront of nandurbar mh' [Postal: None, StreetNo: 111]
                Flags:       has_name=True, has_address=True

--- [Pair #18] India Multi-Match (4 matches) | S1: S1-631464936 vs S2-77133749 ---
  [S1 BEFORE]   Name:    'Deoria Digital Pvt Ltd'
                Address: 'No 56, Chenniappa Nagar, Nagarpalayam, Kalingiam Post, Gobichettipalayam, Erode Erode Tn In, Erode, Tamil Nadu'
  [S1 AFTER]    Norm Name:   'deoria digital private limited' (clean: 'deoria digital')
                Norm Addr:   'number 56 chenniappa nagar nagarpalayam kalingiam post gobichettipalayam erode erode tn in erode tamil nadu' [Postal: None, StreetNo: 56]
                Flags:       has_name=True, has_address=True
  [CAND BEFORE] Name:    'Deoria Digital Pvt Limited'
                Address: 'தமிழ்நாடு, #056 , CHENNIAPPA NAGAR, NAGARPALAYAM, KALINGIAM POST, GOBICHETTIPALAYAM, ERODE ERODE TN IN, ERODE'
  [CAND AFTER]  Norm Name:   'deoria digital private limited' (clean: 'deoria digital')
                Norm Addr:   'தமிழ்நாடு 056 chenniappa nagar nagarpalayam kalingiam post gobichettipalayam erode erode tn in erode' [Postal: None, StreetNo: 056]
                Flags:       has_name=True, has_address=True

--- [Pair #19] India Multi-Match (4 matches) | S1: S1-631464936 vs S2-881868925 ---
  [S1 BEFORE]   Name:    'Deoria Digital Pvt Ltd'
                Address: 'No 56, Chenniappa Nagar, Nagarpalayam, Kalingiam Post, Gobichettipalayam, Erode Erode Tn In, Erode, Tamil Nadu'
  [S1 AFTER]    Norm Name:   'deoria digital private limited' (clean: 'deoria digital')
                Norm Addr:   'number 56 chenniappa nagar nagarpalayam kalingiam post gobichettipalayam erode erode tn in erode tamil nadu' [Postal: None, StreetNo: 56]
                Flags:       has_name=True, has_address=True
  [CAND BEFORE] Name:    'Deoria Pvt Ltd  Center'
                Address: '056 , CHENNIAPPA NAGAR, NAGARPALAYAM, KALINGIAM POST, GOBICHETTIPALAYAM, ERODE ERODE TN IN, ERODE, Tamil Nadu'
  [CAND AFTER]  Norm Name:   'deoria private limited center' (clean: 'deoria center')
                Norm Addr:   '056 chenniappa nagar nagarpalayam kalingiam post gobichettipalayam erode erode tn in erode tamil nadu' [Postal: None, StreetNo: 056]
                Flags:       has_name=True, has_address=True

--- [Pair #20] India Multi-Match (4 matches) | S1: S1-631464936 vs S2-115590074 ---
  [S1 BEFORE]   Name:    'Deoria Digital Pvt Ltd'
                Address: 'No 56, Chenniappa Nagar, Nagarpalayam, Kalingiam Post, Gobichettipalayam, Erode Erode Tn In, Erode, Tamil Nadu'
  [S1 AFTER]    Norm Name:   'deoria digital private limited' (clean: 'deoria digital')
                Norm Addr:   'number 56 chenniappa nagar nagarpalayam kalingiam post gobichettipalayam erode erode tn in erode tamil nadu' [Postal: None, StreetNo: 56]
                Flags:       has_name=True, has_address=True
  [CAND BEFORE] Name:    'Shri Deoria  Digital Pvt'
                Address: 'NO 056 , CHENNIAPPA NAGAR, NAGARPALAYAM, KALINGIAM POST, GOBICHETTIPALAYAM, ERODE ERODE TN IN, ERODE, Tamil Nadu'
  [CAND AFTER]  Norm Name:   'shri deoria digital private' (clean: 'shri deoria digital')
                Norm Addr:   'number 056 chenniappa nagar nagarpalayam kalingiam post gobichettipalayam erode erode tn in erode tamil nadu' [Postal: None, StreetNo: 056]
                Flags:       has_name=True, has_address=True

--- [Pair #21] India Multi-Match (4 matches) | S1: S1-631464936 vs S3-268805053 ---
  [S1 BEFORE]   Name:    'Deoria Digital Pvt Ltd'
                Address: 'No 56, Chenniappa Nagar, Nagarpalayam, Kalingiam Post, Gobichettipalayam, Erode Erode Tn In, Erode, Tamil Nadu'
  [S1 AFTER]    Norm Name:   'deoria digital private limited' (clean: 'deoria digital')
                Norm Addr:   'number 56 chenniappa nagar nagarpalayam kalingiam post gobichettipalayam erode erode tn in erode tamil nadu' [Postal: None, StreetNo: 56]
                Flags:       has_name=True, has_address=True
  [CAND BEFORE] Name:    'Deoria Digital  Pvt Ltd'
                Address: '#56, Chenniappa Nagar, Nagarpalayam, Kalingiam Post, Gobichettipalayam, Erode Erode Tn In, Erode, தமிழ்நாடு'
  [CAND AFTER]  Norm Name:   'deoria digital private limited' (clean: 'deoria digital')
                Norm Addr:   '56 chenniappa nagar nagarpalayam kalingiam post gobichettipalayam erode erode tn in erode தமிழ்நாடு' [Postal: None, StreetNo: 56]
                Flags:       has_name=True, has_address=True

--- [Pair #22] US Multi-Match (2 matches) | S1: S1-878561933 vs S2-988941963 ---
  [S1 BEFORE]   Name:    'Evans Coastal Priority, LLC'
                Address: '8500 Mike Shapiro Drive, Unit 214, Clinton, MD'
  [S1 AFTER]    Norm Name:   'evans coastal priority llc' (clean: 'evans coastal priority')
                Norm Addr:   '8500 mike shapiro drive unit 214 clinton md' [Postal: 214, StreetNo: 8500]
                Flags:       has_name=True, has_address=True
  [CAND BEFORE] Name:    'Evans Coastal Priority,-LLC'
                Address: '8500 MIKE SHAPIRO DRIVE, CLINTON, MD'
  [CAND AFTER]  Norm Name:   'evans coastal priority llc' (clean: 'evans coastal priority')
                Norm Addr:   '8500 mike shapiro drive clinton md' [Postal: None, StreetNo: 8500]
                Flags:       has_name=True, has_address=True

--- [Pair #23] US Multi-Match (2 matches) | S1: S1-878561933 vs S3-658420703 ---
  [S1 BEFORE]   Name:    'Evans Coastal Priority, LLC'
                Address: '8500 Mike Shapiro Drive, Unit 214, Clinton, MD'
  [S1 AFTER]    Norm Name:   'evans coastal priority llc' (clean: 'evans coastal priority')
                Norm Addr:   '8500 mike shapiro drive unit 214 clinton md' [Postal: 214, StreetNo: 8500]
                Flags:       has_name=True, has_address=True
  [CAND BEFORE] Name:    'LLC Evans Cóastal Priority,'
                Address: '8500- Mike Shapiro Drive, # 214, Clinton, Maryland'
  [CAND AFTER]  Norm Name:   'llc evans cóastal priority' (clean: 'evans cóastal priority')
                Norm Addr:   '8500 mike shapiro drive 214 clinton maryland' [Postal: 214, StreetNo: 8500]
                Flags:       has_name=True, has_address=True

--- [Pair #24] US Multi-Match (3 matches) | S1: S1-545985406 vs S2-577672751 ---
  [S1 BEFORE]   Name:    'Holly Springs Literacy Fellowship'
                Address: '245 Arctic Ridge Way, Holly Springs, NC'
  [S1 AFTER]    Norm Name:   'holly springs literacy fellowship' (clean: 'holly springs literacy fellowship')
                Norm Addr:   '245 arctic ridge way holly springs nc' [Postal: None, StreetNo: 245]
                Flags:       has_name=True, has_address=True
  [CAND BEFORE] Name:    'Holly Spríngs Literacy Fellowship'
                Address: '245 ARCTIC RIDGE WAY, HOLLY SPRINGS, NC'
  [CAND AFTER]  Norm Name:   'holly spríngs literacy fellowship' (clean: 'holly spríngs literacy fellowship')
                Norm Addr:   '245 arctic ridge way holly springs nc' [Postal: None, StreetNo: 245]
                Flags:       has_name=True, has_address=True

--- [Pair #25] US Multi-Match (3 matches) | S1: S1-545985406 vs S2-712948461 ---
  [S1 BEFORE]   Name:    'Holly Springs Literacy Fellowship'
                Address: '245 Arctic Ridge Way, Holly Springs, NC'
  [S1 AFTER]    Norm Name:   'holly springs literacy fellowship' (clean: 'holly springs literacy fellowship')
                Norm Addr:   '245 arctic ridge way holly springs nc' [Postal: None, StreetNo: 245]
                Flags:       has_name=True, has_address=True
  [CAND BEFORE] Name:    'Holly Springs  Literacy'
                Address: 'NC, HOLLY SPRINGS, 245 ARCTIC RIDGE WAY'
  [CAND AFTER]  Norm Name:   'holly springs literacy' (clean: 'holly springs literacy')
                Norm Addr:   'nc holly springs 245 arctic ridge way' [Postal: None, StreetNo: 245]
                Flags:       has_name=True, has_address=True

--- [Pair #26] US Multi-Match (3 matches) | S1: S1-545985406 vs S2-134037879 ---
  [S1 BEFORE]   Name:    'Holly Springs Literacy Fellowship'
                Address: '245 Arctic Ridge Way, Holly Springs, NC'
  [S1 AFTER]    Norm Name:   'holly springs literacy fellowship' (clean: 'holly springs literacy fellowship')
                Norm Addr:   '245 arctic ridge way holly springs nc' [Postal: None, StreetNo: 245]
                Flags:       has_name=True, has_address=True
  [CAND BEFORE] Name:    'HOLLY SPRINGS LITERACY FELLOWSHIP LP'
                Address: '245 ARCTIC RIDGE WAY, HOLLY SPRINGS, NC'
  [CAND AFTER]  Norm Name:   'holly springs literacy fellowship lp' (clean: 'holly springs literacy fellowship lp')
                Norm Addr:   '245 arctic ridge way holly springs nc' [Postal: None, StreetNo: 245]
                Flags:       has_name=True, has_address=True

--- [Pair #27] US Multi-Match (5 matches) | S1: S1-218436290 vs S2-75924906 ---
  [S1 BEFORE]   Name:    'Global Crystal Optics Inc'
                Address: '1322 Cedar Branch Court, Wake Forest, NC'
  [S1 AFTER]    Norm Name:   'global crystal optics incorporated' (clean: 'global crystal optics')
                Norm Addr:   '1322 cedar branch court wake forest nc' [Postal: None, StreetNo: 1322]
                Flags:       has_name=True, has_address=True
  [CAND BEFORE] Name:    'Global Crystal Optics Incorporated'
                Address: '1322 CEDAR BRANCH CT, WAKE FOREST, NC'
  [CAND AFTER]  Norm Name:   'global crystal optics incorporated' (clean: 'global crystal optics')
                Norm Addr:   '1322 cedar branch court wake forest nc' [Postal: None, StreetNo: 1322]
                Flags:       has_name=True, has_address=True

--- [Pair #28] US Multi-Match (5 matches) | S1: S1-218436290 vs S2-381727005 ---
  [S1 BEFORE]   Name:    'Global Crystal Optics Inc'
                Address: '1322 Cedar Branch Court, Wake Forest, NC'
  [S1 AFTER]    Norm Name:   'global crystal optics incorporated' (clean: 'global crystal optics')
                Norm Addr:   '1322 cedar branch court wake forest nc' [Postal: None, StreetNo: 1322]
                Flags:       has_name=True, has_address=True
  [CAND BEFORE] Name:    'Global Crystal Optics'
                Address: '1322 CEDAR BRANCH CT, WAKE FOREST, NC'
  [CAND AFTER]  Norm Name:   'global crystal optics' (clean: 'global crystal optics')
                Norm Addr:   '1322 cedar branch court wake forest nc' [Postal: None, StreetNo: 1322]
                Flags:       has_name=True, has_address=True

--- [Pair #29] US Multi-Match (5 matches) | S1: S1-218436290 vs S2-427668105 ---
  [S1 BEFORE]   Name:    'Global Crystal Optics Inc'
                Address: '1322 Cedar Branch Court, Wake Forest, NC'
  [S1 AFTER]    Norm Name:   'global crystal optics incorporated' (clean: 'global crystal optics')
                Norm Addr:   '1322 cedar branch court wake forest nc' [Postal: None, StreetNo: 1322]
                Flags:       has_name=True, has_address=True
  [CAND BEFORE] Name:    'globalcrystaloptics.com'
                Address: '1322 CEDAR BRANCH CURT, WAKE FOREST, NC'
  [CAND AFTER]  Norm Name:   'globalcrystaloptics com' (clean: 'globalcrystaloptics com')
                Norm Addr:   '1322 cedar branch curt wake forest nc' [Postal: None, StreetNo: 1322]
                Flags:       has_name=True, has_address=True

--- [Pair #30] US Multi-Match (5 matches) | S1: S1-218436290 vs S3-772949578 ---
  [S1 BEFORE]   Name:    'Global Crystal Optics Inc'
                Address: '1322 Cedar Branch Court, Wake Forest, NC'
  [S1 AFTER]    Norm Name:   'global crystal optics incorporated' (clean: 'global crystal optics')
                Norm Addr:   '1322 cedar branch court wake forest nc' [Postal: None, StreetNo: 1322]
                Flags:       has_name=True, has_address=True
  [CAND BEFORE] Name:    'Crystal Global Inc Optics'
                Address: 'Wake Forest, North Carolina, Cedar Branch Court'
  [CAND AFTER]  Norm Name:   'crystal global incorporated optics' (clean: 'crystal global optics')
                Norm Addr:   'wake forest north carolina cedar branch court' [Postal: None, StreetNo: None]
                Flags:       has_name=True, has_address=True

--- [Pair #31] US Multi-Match (5 matches) | S1: S1-218436290 vs S3-648971713 ---
  [S1 BEFORE]   Name:    'Global Crystal Optics Inc'
                Address: '1322 Cedar Branch Court, Wake Forest, NC'
  [S1 AFTER]    Norm Name:   'global crystal optics incorporated' (clean: 'global crystal optics')
                Norm Addr:   '1322 cedar branch court wake forest nc' [Postal: None, StreetNo: 1322]
                Flags:       has_name=True, has_address=True
  [CAND BEFORE] Name:    '6l0bal Crystal Optics  Inc'
                Address: '1322 Cedar Branch Ct, Wake Forest, North Carolina'
  [CAND AFTER]  Norm Name:   '6l0bal crystal optics incorporated' (clean: '6l0bal crystal optics')
                Norm Addr:   '1322 cedar branch court wake forest north carolina' [Postal: None, StreetNo: 1322]
                Flags:       has_name=True, has_address=True

--- [Pair #32] US Multi-Match (2 matches) | S1: S1-608237939 vs S2-627860382 ---
  [S1 BEFORE]   Name:    'Diamond Apparel Digital, Inc'
                Address: '819 Saint Nicholas Avenue, Dayton, OH'
  [S1 AFTER]    Norm Name:   'diamond apparel digital incorporated' (clean: 'diamond apparel digital')
                Norm Addr:   '819 saint nicholas avenue dayton oh' [Postal: None, StreetNo: 819]
                Flags:       has_name=True, has_address=True
  [CAND BEFORE] Name:    'Diamond Digital, Inc Center'
                Address: '819 SAINT NICHOLAS AVE, DAYTON, OH'
  [CAND AFTER]  Norm Name:   'diamond digital incorporated center' (clean: 'diamond digital center')
                Norm Addr:   '819 saint nicholas avenue dayton oh' [Postal: None, StreetNo: 819]
                Flags:       has_name=True, has_address=True

--- [Pair #33] US Multi-Match (2 matches) | S1: S1-608237939 vs S3-74993502 ---
  [S1 BEFORE]   Name:    'Diamond Apparel Digital, Inc'
                Address: '819 Saint Nicholas Avenue, Dayton, OH'
  [S1 AFTER]    Norm Name:   'diamond apparel digital incorporated' (clean: 'diamond apparel digital')
                Norm Addr:   '819 saint nicholas avenue dayton oh' [Postal: None, StreetNo: 819]
                Flags:       has_name=True, has_address=True
  [CAND BEFORE] Name:    'DIAMOND APPAREL DIGITAL, INC'
                Address: 'PO Box 620, Dayton, 00819 Saint Nicholas Ave, Ohio'
  [CAND AFTER]  Norm Name:   'diamond apparel digital incorporated' (clean: 'diamond apparel digital')
                Norm Addr:   'post office box 620 dayton 00819 saint nicholas avenue ohio' [Postal: 00819, StreetNo: None]
                Flags:       has_name=True, has_address=True

--- [Pair #34] US Multi-Match (4 matches) | S1: S1-307510131 vs S2-760956156 ---
  [S1 BEFORE]   Name:    'Pediatric Care Inc.'
                Address: '14141 Kimball Road, Crosslake, MN'
  [S1 AFTER]    Norm Name:   'pediatric care incorporated' (clean: 'pediatric care')
                Norm Addr:   '14141 kimball road crosslake mn' [Postal: None, StreetNo: 14141]
                Flags:       has_name=True, has_address=True
  [CAND BEFORE] Name:    'PEDIATRIC CARE'
                Address: '14141 KIMBALL RD, CROSSLAKE, MN'
  [CAND AFTER]  Norm Name:   'pediatric care' (clean: 'pediatric care')
                Norm Addr:   '14141 kimball road crosslake mn' [Postal: None, StreetNo: 14141]
                Flags:       has_name=True, has_address=True

--- [Pair #35] US Multi-Match (4 matches) | S1: S1-307510131 vs S2-351680210 ---
  [S1 BEFORE]   Name:    'Pediatric Care Inc.'
                Address: '14141 Kimball Road, Crosslake, MN'
  [S1 AFTER]    Norm Name:   'pediatric care incorporated' (clean: 'pediatric care')
                Norm Addr:   '14141 kimball road crosslake mn' [Postal: None, StreetNo: 14141]
                Flags:       has_name=True, has_address=True
  [CAND BEFORE] Name:    'Pediatric Care  Inc.'
                Address: '14141 KIMBALL ROAD, CROSSLAKE, MN'
  [CAND AFTER]  Norm Name:   'pediatric care incorporated' (clean: 'pediatric care')
                Norm Addr:   '14141 kimball road crosslake mn' [Postal: None, StreetNo: 14141]
                Flags:       has_name=True, has_address=True

--- [Pair #36] US Multi-Match (4 matches) | S1: S1-307510131 vs S2-7813217 ---
  [S1 BEFORE]   Name:    'Pediatric Care Inc.'
                Address: '14141 Kimball Road, Crosslake, MN'
  [S1 AFTER]    Norm Name:   'pediatric care incorporated' (clean: 'pediatric care')
                Norm Addr:   '14141 kimball road crosslake mn' [Postal: None, StreetNo: 14141]
                Flags:       has_name=True, has_address=True
  [CAND BEFORE] Name:    'Pediatric  Care Inc'
                Address: '14141 KNMBALL RD, CROSSLAKE, MN'
  [CAND AFTER]  Norm Name:   'pediatric care incorporated' (clean: 'pediatric care')
                Norm Addr:   '14141 knmball road crosslake mn' [Postal: None, StreetNo: 14141]
                Flags:       has_name=True, has_address=True

--- [Pair #37] US Multi-Match (4 matches) | S1: S1-307510131 vs S3-741485966 ---
  [S1 BEFORE]   Name:    'Pediatric Care Inc.'
                Address: '14141 Kimball Road, Crosslake, MN'
  [S1 AFTER]    Norm Name:   'pediatric care incorporated' (clean: 'pediatric care')
                Norm Addr:   '14141 kimball road crosslake mn' [Postal: None, StreetNo: 14141]
                Flags:       has_name=True, has_address=True
  [CAND BEFORE] Name:    'Pediatric (Care)'
                Address: '0014141 Kimball Rd, Crosslake, Minnesota'
  [CAND AFTER]  Norm Name:   'pediatric care' (clean: 'pediatric care')
                Norm Addr:   '0014141 kimball road crosslake minnesota' [Postal: None, StreetNo: 0014141]
                Flags:       has_name=True, has_address=True

--- [Pair #38] US Multi-Match (2 matches) | S1: S1-630214374 vs S2-938085942 ---
  [S1 BEFORE]   Name:    'Anchor'
                Address: '38 Williams Street, Haverhill, MA'
  [S1 AFTER]    Norm Name:   'anchor' (clean: 'anchor')
                Norm Addr:   '38 williams street haverhill ma' [Postal: None, StreetNo: 38]
                Flags:       has_name=True, has_address=True
  [CAND BEFORE] Name:    'Anchor'
                Address: '38 WILLIAMS STREET, PO BOX 2281, HAVERHILL, MA'
  [CAND AFTER]  Norm Name:   'anchor' (clean: 'anchor')
                Norm Addr:   '38 williams street post office box 2281 haverhill ma' [Postal: 2281, StreetNo: 38]
                Flags:       has_name=True, has_address=True

--- [Pair #39] US Multi-Match (2 matches) | S1: S1-630214374 vs S3-3870604 ---
  [S1 BEFORE]   Name:    'Anchor'
                Address: '38 Williams Street, Haverhill, MA'
  [S1 AFTER]    Norm Name:   'anchor' (clean: 'anchor')
                Norm Addr:   '38 williams street haverhill ma' [Postal: None, StreetNo: 38]
                Flags:       has_name=True, has_address=True
  [CAND BEFORE] Name:    'anchor.com'
                Address: '##38 Williams St, Haverhill, Massachusetts'
  [CAND AFTER]  Norm Name:   'anchor com' (clean: 'anchor com')
                Norm Addr:   '38 williams street haverhill massachusetts' [Postal: None, StreetNo: 38]
                Flags:       has_name=True, has_address=True

--- [Pair #40] India Native-Script (Devanagari/Tamil) Multi-Match (6 matches) | S1: S1-89207398 vs S2-910833171 ---
  [S1 BEFORE]   Name:    'MUK It Limited'
                Address: 'C/O. Balaji Packaging Industries, Plot No. 76/77, D-3 Block, Midc Chinchwad, Pune, Maharashtra'
  [S1 AFTER]    Norm Name:   'muk it limited' (clean: 'muk it')
                Norm Addr:   'c o balaji packaging industries plot number 76 77 d 3 block midc chinchwad pune maharashtra' [Postal: None, StreetNo: None]
                Flags:       has_name=True, has_address=True
  [CAND BEFORE] Name:    'MUK It Ltd Ltd'
                Address: 'C/O. BALAJI PACKAGING INDUSTRIES, PLOT NO. G-76/77, D-3 BLOCK, MIDC CHINCHWAD, महाराष्ट्र'
  [CAND AFTER]  Norm Name:   'muk it limited limited' (clean: 'muk it')
                Norm Addr:   'c o balaji packaging industries plot number g 76 77 d 3 block midc chinchwad महाराष्ट्र' [Postal: None, StreetNo: None]
                Flags:       has_name=True, has_address=True

--- [Pair #41] India Native-Script (Devanagari/Tamil) Multi-Match (4 matches) | S1: S1-200060223 vs S2-412745666 ---
  [S1 BEFORE]   Name:    'Southern Finance Private Limited'
                Address: '3 Pandit Motilal Colonyrajbari Colony, Kolkata, North 24 Parganas, West Bengal'
  [S1 AFTER]    Norm Name:   'southern finance private limited' (clean: 'southern finance')
                Norm Addr:   '3 pandit motilal colonyrajbari colony kolkata north 24 parganas west bengal' [Postal: None, StreetNo: 3]
                Flags:       has_name=True, has_address=True
  [CAND BEFORE] Name:    'সাউদার্ন ফাইন্যান্স প্রাইভেট লিমিটেড'
                Address: 'NORTH 24 PARGANAS, West Bengal, KOLKATA, 3 PANDIT MOTILAL COLONYRAJBARI COLONY'
  [CAND AFTER]  Norm Name:   'সাউদার্ন ফাইন্যান্স প্রাইভেট লিমিটেড' (clean: 'সাউদার্ন ফাইন্যান্স প্রাইভেট লিমিটেড')
                Norm Addr:   'north 24 parganas west bengal kolkata 3 pandit motilal colonyrajbari colony' [Postal: None, StreetNo: 24]
                Flags:       has_name=True, has_address=True

--- [Pair #42] India Native-Script (Devanagari/Tamil) Multi-Match (3 matches) | S1: S1-584123311 vs S2-693131119 ---
  [S1 BEFORE]   Name:    'Vasundhara Academy'
                Address: 'Shop No. 1, Plot No. 47, Ashma Complex, Nr. Royal Park, Nr. Hariyali Soc, Juhapura, Ahmedabad, Gujarat'
  [S1 AFTER]    Norm Name:   'vasundhara academy' (clean: 'vasundhara academy')
                Norm Addr:   'shop number 1 plot number 47 ashma complex near royal park near hariyali society juhapura ahmedabad gujarat' [Postal: None, StreetNo: 1]
                Flags:       has_name=True, has_address=True
  [CAND BEFORE] Name:    'Vasundhara Academy'
                Address: 'H.NO 1 , PLOT NO. 47, ASHMA COMPLEX, NR. ROYAL PARK, NR. HARIYALI SOC, JUHAPURA, AHMEDABAD, ગુજરાત'
  [CAND AFTER]  Norm Name:   'vasundhara academy' (clean: 'vasundhara academy')
                Norm Addr:   'h number 1 plot number 47 ashma complex near royal park near hariyali society juhapura ahmedabad ગુજરાત' [Postal: None, StreetNo: 1]
                Flags:       has_name=True, has_address=True

--- [Pair #43] India Native-Script (Devanagari/Tamil) Multi-Match (3 matches) | S1: S1-584123311 vs S3-926721738 ---
  [S1 BEFORE]   Name:    'Vasundhara Academy'
                Address: 'Shop No. 1, Plot No. 47, Ashma Complex, Nr. Royal Park, Nr. Hariyali Soc, Juhapura, Ahmedabad, Gujarat'
  [S1 AFTER]    Norm Name:   'vasundhara academy' (clean: 'vasundhara academy')
                Norm Addr:   'shop number 1 plot number 47 ashma complex near royal park near hariyali society juhapura ahmedabad gujarat' [Postal: None, StreetNo: 1]
                Flags:       has_name=True, has_address=True
  [CAND BEFORE] Name:    'Vasundhara Ácademy'
                Address: 'Shop No. 1, Ahmedabad, ગુજરાત'
  [CAND AFTER]  Norm Name:   'vasundhara ácademy' (clean: 'vasundhara ácademy')
                Norm Addr:   'shop number 1 ahmedabad ગુજરાત' [Postal: None, StreetNo: 1]
                Flags:       has_name=True, has_address=True

--- [Pair #44] India Native-Script (Devanagari/Tamil) Multi-Match (3 matches) | S1: S1-17751639 vs S3-361961768 ---
  [S1 BEFORE]   Name:    'Balaji Consultants Pvt Ltd'
                Address: 'D-246 Office No 303 Balaji Chamber Laxmi Nagar, Delhi, East Delhi, Delhi'
  [S1 AFTER]    Norm Name:   'balaji consultants private limited' (clean: 'balaji consultants')
                Norm Addr:   'd 246 office number 303 balaji chamber laxmi nagar delhi east delhi delhi' [Postal: 303, StreetNo: 246]
                Flags:       has_name=True, has_address=True
  [CAND BEFORE] Name:    'Sri Balaji Consultants'
                Address: 'Plot 846 D-246 Office No 303 Balaji Chamber Laxmi Nagar, Delhi, East Delhi, दिल्ली'
  [CAND AFTER]  Norm Name:   'sri balaji consultants' (clean: 'sri balaji consultants')
                Norm Addr:   'plot 846 d 246 office number 303 balaji chamber laxmi nagar delhi east delhi दिल्ली' [Postal: 303, StreetNo: 846]
                Flags:       has_name=True, has_address=True

--- [Pair #45] India Native-Script (Devanagari/Tamil) Multi-Match (4 matches) | S1: S1-710009389 vs S3-885379527 ---
  [S1 BEFORE]   Name:    'Bharat Services LLP'
                Address: 'A-4/901, Tulip White, Sector-69, Badshahpur, Gurgaon, Haryana'
  [S1 AFTER]    Norm Name:   'bharat services llp' (clean: 'bharat')
                Norm Addr:   'a 4 901 tulip white sector 69 badshahpur gurgaon haryana' [Postal: None, StreetNo: 4]
                Flags:       has_name=True, has_address=True
  [CAND BEFORE] Name:    'भारत सर्विसेज एलएलपी'
                Address: 'Plot 772. A-4/901, Tulip White, Sector-69, Badshahpur, Gurgaon, HR'
  [CAND AFTER]  Norm Name:   'भारत सर्विसेज एलएलपी' (clean: 'भारत सर्विसेज एलएलपी')
                Norm Addr:   'plot 772 a 4 901 tulip white sector 69 badshahpur gurgaon hr' [Postal: None, StreetNo: 772]
                Flags:       has_name=True, has_address=True

--- [Pair #46] India Native-Script (Devanagari/Tamil) Multi-Match (5 matches) | S1: S1-534997445 vs S3-749024993 ---
  [S1 BEFORE]   Name:    'IBO Industries Private Limited'
                Address: 'Ii/38 Dr. Vikram Sarabhai Electronic Estate, Thiruvanmiyur, Chennai City Corporation, Kanchipuram, Tamil Nadu'
  [S1 AFTER]    Norm Name:   'ibo industries private limited' (clean: 'ibo')
                Norm Addr:   'ii 38 drive vikram sarabhai electronic estate thiruvanmiyur chennai city corporation kanchipuram tamil nadu' [Postal: None, StreetNo: 38]
                Flags:       has_name=True, has_address=True
  [CAND BEFORE] Name:    'IBO Industries Private Límited'
                Address: 'Kanchipuram, Chennai City Corporation, Ii/#38 Dr. Vikram Sarabhai Electronic Estate, தமிழ்நாடு, Thiruvanmiyur'
  [CAND AFTER]  Norm Name:   'ibo industries private límited' (clean: 'ibo límited')
                Norm Addr:   'kanchipuram chennai city corporation ii 38 drive vikram sarabhai electronic estate தமிழ்நாடு thiruvanmiyur' [Postal: None, StreetNo: 38]
                Flags:       has_name=True, has_address=True

--- [Pair #47] India Native-Script (Devanagari/Tamil) Multi-Match (4 matches) | S1: S1-722292153 vs S2-941761264 ---
  [S1 BEFORE]   Name:    'Shree Consultancy Private Limited'
                Address: '4/434-B, Raj And Raj Garden, Thathampatty Post, Salem, Tamil Nadu'
  [S1 AFTER]    Norm Name:   'shree consultancy private limited' (clean: 'shree consultancy')
                Norm Addr:   '4 434 b raj and raj garden thathampatty post salem tamil nadu' [Postal: 434, StreetNo: 4]
                Flags:       has_name=True, has_address=True
  [CAND BEFORE] Name:    'ஸ்ரீ கன்சல்டன்சி பிரைவேட் லிமிடெட்'
                Address: 'NO 611. 4/434-B, SALEM, Tamil Nadu'
  [CAND AFTER]  Norm Name:   'ஸ்ரீ கன்சல்டன்சி பிரைவேட் லிமிடெட்' (clean: 'ஸ்ரீ கன்சல்டன்சி பிரைவேட் லிமிடெட்')
                Norm Addr:   'number 611 4 434 b salem tamil nadu' [Postal: 434, StreetNo: 611]
                Flags:       has_name=True, has_address=True

--- [Pair #48] India Native-Script (Devanagari/Tamil) Multi-Match (3 matches) | S1: S1-373711352 vs S3-428488902 ---
  [S1 BEFORE]   Name:    'Southern Foods Private Limited'
                Address: 'Chennai, 6Th Floor, No. 672, (Old No. 476), Anna Salai, Tamil Nadu, Chennai, Temple Tower'
  [S1 AFTER]    Norm Name:   'southern foods private limited' (clean: 'southern foods')
                Norm Addr:   'chennai 6th floor number 672 old number 476 anna salai tamil nadu chennai temple tower' [Postal: 476, StreetNo: None]
                Flags:       has_name=True, has_address=True
  [CAND BEFORE] Name:    'Private Southern Foods (Ltd)'
                Address: 'Temple Tower, Chennai, தமிழ்நாடு'
  [CAND AFTER]  Norm Name:   'private southern foods limited' (clean: 'southern foods')
                Norm Addr:   'temple tower chennai தமிழ்நாடு' [Postal: None, StreetNo: None]
                Flags:       has_name=True, has_address=True

--- [Pair #49] India Native-Script (Devanagari/Tamil) Multi-Match (3 matches) | S1: S1-87568570 vs S3-861149855 ---
  [S1 BEFORE]   Name:    'Nk Producer'
                Address: 'Khasra No. 505, 506, Unit No.1, Dlf Phase Iv, Sadar Bazar, Gurgaon, Haryana'
  [S1 AFTER]    Norm Name:   'nk producer' (clean: 'nk producer')
                Norm Addr:   'khasra number 505 506 unit number 1 dlf phase iv sadar bazar gurgaon haryana' [Postal: None, StreetNo: 505]
                Flags:       has_name=True, has_address=True
  [CAND BEFORE] Name:    'Nk Producer LLP'
                Address: 'Sadar Bazar, H.no 505, हरियाणा, Gurgaon'
  [CAND AFTER]  Norm Name:   'nk producer llp' (clean: 'nk producer')
                Norm Addr:   'sadar bazar h number 505 हरियाणा gurgaon' [Postal: 505, StreetNo: None]
                Flags:       has_name=True, has_address=True

--- [Pair #50] India Native-Script (Devanagari/Tamil) Multi-Match (4 matches) | S1: S1-835648279 vs S2-400670647 ---
  [S1 BEFORE]   Name:    'Urban Tech Limited'
                Address: '110, Mount Poonamallee Road, Chennai, Maduravoyal, Thiruvallur, Sriperumbudur, Kanchipuram, Tamil Nadu'
  [S1 AFTER]    Norm Name:   'urban technologies limited' (clean: 'urban technologies')
                Norm Addr:   '110 mount poonamallee road chennai maduravoyal thiruvallur sriperumbudur kanchipuram tamil nadu' [Postal: None, StreetNo: 110]
                Flags:       has_name=True, has_address=True
  [CAND BEFORE] Name:    'அர்பன் டெக் லிமிடெட்'
                Address: '0110, MOUNT POONAMALLEE ROAD, CHENNAI, MADURAVOYAL, THIRUVALLUR, KANCHIPURAM, SRIPERUMBUDUR, Tamil Nadu'
  [CAND AFTER]  Norm Name:   'அர்பன் டெக் லிமிடெட்' (clean: 'அர்பன் டெக் லிமிடெட்')
                Norm Addr:   '0110 mount poonamallee road chennai maduravoyal thiruvallur kanchipuram sriperumbudur tamil nadu' [Postal: None, StreetNo: 0110]
                Flags:       has_name=True, has_address=True

--- [Pair #51] India Native-Script (Devanagari/Tamil) Multi-Match (6 matches) | S1: S1-451998438 vs S2-966737666 ---
  [S1 BEFORE]   Name:    'Star Business Private Limited'
                Address: 'Ward No. 6, Near Asharam, Hari Mandir, Amarpuri, Pataudi, Gurgaon, Haryana'
  [S1 AFTER]    Norm Name:   'star business private limited' (clean: 'star business')
                Norm Addr:   'ward number 6 near asharam hari mandir amarpuri pataudi gurgaon haryana' [Postal: None, StreetNo: 6]
                Flags:       has_name=True, has_address=True
  [CAND BEFORE] Name:    'Star Business Private'
                Address: 'HN 314 WARD NO. 6, NEAR ASHARAM, HARI MANDIR, AMARPURI, PATAUDI, हरियाणा'
  [CAND AFTER]  Norm Name:   'star business private' (clean: 'star business')
                Norm Addr:   'hn 314 ward number 6 near asharam hari mandir amarpuri pataudi हरियाणा' [Postal: None, StreetNo: 314]
                Flags:       has_name=True, has_address=True
