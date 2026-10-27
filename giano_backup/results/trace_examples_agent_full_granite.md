# Fixed chain (baseline_vanilla_rag_v3) vs dynamic agent (agent_full_granite)

Questions where EM differs: agent better 16, agent worse 123 (of 500).


## agent BETTER than the fixed chain

### qid 97  (category 2)
**Question:** Which kind of cancerous growth did researchers implant in mice to evaluate the tumor-inhibiting effects of Livistona chinensis extract?

**Gold:** Fibrosarcoma

**Fixed chain:** pred = `Glass cylinder`  (EM 0), gold paper rank 1

**Agent:** pred = `Fibrosarcoma`  (EM 1); stop = final; steps = 2; tools = {'retrieve_from_index': 1}; gold paper seen = True

- step 1 [action] `retrieve_from_index(cancerous growth Livistona chinensis extract mouse study)`  — thought: I should search the local PubMed corpus for information on the type of cancerous growth used in the study.  → obs: '[1] Livistona extract inhibits angiogenesis and cancer growth.\nIn a screen for naturally o'
- step 2 [final]  — thought: I now know the final answer

*Auto notes:* none

*ANNOTATION (fill in by hand):* where did the agent decide differently, and was it better or worse?

### qid 327  (category 4)
**Question:** What laboratory technique is commonly used to determine the serotype of meningococcal strains?

**Gold:** Serological typing

**Fixed chain:** pred = `Serological typing technique`  (EM 0), gold paper rank 1

**Agent:** pred = `Serotyping`  (EM 1); stop = final; steps = 4; tools = {'query_reformulator': 1, 'generate_reasoning_query': 1, 'fetch_pubmed_live': 1}; gold paper seen = False

- step 1 [action] `query_reformulator(determine the serotype of meningococcal strains)`  — thought: I should reformulate the question into a keyword-style search query.  → obs: 'Search query: serotype of meningococcal strains'
- step 2 [action] `generate_reasoning_query(determine the serotype of meningococcal strains)`  — thought: I should generate a reasoning query to guide the search.  → obs: 'Reasoning (use it as the query for retrieve_from_index): The question requires identifying'
- step 3 [action] `fetch_pubmed_live(serotyping meningococcal)`  — thought: I should fetch specific keywords from PubMed to find relevant abstracts.  → obs: '[1] Community-acquired bacterial meningitis in adults in the Netherlands, 2006-14: a prosp'
- step 4 [final]  — thought: I now know the final answer

*Auto notes:* gold paper never retrieved

*ANNOTATION (fill in by hand):* where did the agent decide differently, and was it better or worse?

### qid 136  (category 4)
**Question:** Which gene is frequently employed as an internal reference to evaluate specimen integrity during PCR assays aimed at detecting viral DNA in human tissue samples?

**Gold:** GAPDH

**Fixed chain:** pred = `β-actin`  (EM 0), gold paper rank 38

**Agent:** pred = `GAPDH`  (EM 1); stop = final; steps = 3; tools = {'query_reformulator': 1, 'generate_reasoning_query': 1}; gold paper seen = False

- step 1 [action] `query_reformulator(Which gene is frequently employed as an internal reference to evaluate specimen )`  — thought: I should first reformulate the question into a keyword-style search query.  → obs: 'Search query: gene internal reference PCR specimen integrity viral DNA human tissue sample'
- step 2 [action] `generate_reasoning_query(Which gene is frequently employed as an internal reference to evaluate specimen )`  — thought: I should generate a reasoning query to guide the search.  → obs: 'Reasoning (use it as the query for retrieve_from_index): The question asks about a specifi'
- step 3 [final]  — thought: I now know the final answer

*Auto notes:* gold paper never retrieved

*ANNOTATION (fill in by hand):* where did the agent decide differently, and was it better or worse?


## agent WORSE than the fixed chain

### qid 483  (category 2)
**Question:** Which selenium-containing molecule is identified as a substrate for glutathione disulfide reductase, which may lead to an overestimation of its antioxidant capacity?

**Gold:** Ebselen

**Fixed chain:** pred = `ebselen`  (EM 1), gold paper rank 1

**Agent:** pred = `Selenocysteine`  (EM 0); stop = final; steps = 4; tools = {'query_reformulator': 1, 'generate_reasoning_query': 1, 'fetch_pubmed_live': 1}; gold paper seen = False

- step 1 [action] `query_reformulator(selenium-containing molecule substrate glutathione disulfide reductase antioxida)`  — thought: I should first reformulate the question into a keyword-style search query.  → obs: 'Search query: selenium-containing molecule substrate glutathione disulfide reductase antio'
- step 2 [action] `generate_reasoning_query(selenium-containing molecule substrate glutathione disulfide reductase antioxida)`  — thought: I should generate a reasoning query to guide the search.  → obs: 'Reasoning (use it as the query for retrieve_from_index): The question links selenium-conta'
- step 3 [action] `fetch_pubmed_live(selenium glutathione disulfide reductase)`  — thought: I should now search PubMed for the specific molecule.  → obs: '[1] Selenium.\nSelenium occurs as inorganic selenite or selenate and in organic forms in pl'
- step 4 [final]  — thought: The abstracts discuss selenium-containing molecules and glutathione reductase, indicating the molecule in question.

*Auto notes:* gold paper never retrieved

*ANNOTATION (fill in by hand):* where did the agent decide differently, and was it better or worse?

### qid 451  (category 2)
**Question:** Which naturally occurring compound has demonstrated an ability to enhance the efficacy of cisplatin treatment for esophageal squamous cell carcinoma through its influence on intestinal microbial communities?

**Gold:** Cepharanthine hydrochloride

**Fixed chain:** pred = `Cepharanthine hydrochloride`  (EM 1), gold paper rank 1

**Agent:** pred = `Fecal microbiota transplantation`  (EM 0); stop = final; steps = 3; tools = {'query_reformulator': 1, 'generate_reasoning_query': 1}; gold paper seen = False

- step 1 [action] `query_reformulator(Which naturally occurring compound has demonstrated an ability to enhance the ef)`  — thought: I should first reformulate the question into a keyword-style search query.  → obs: 'Search query: naturally occurring compound, enhance cisplatin efficacy, esophageal squamou'
- step 2 [action] `generate_reasoning_query(Which naturally occurring compound has demonstrated an ability to enhance the ef)`  — thought: I should generate a reasoning query to guide the search.  → obs: 'Reasoning (use it as the query for retrieve_from_index): The question asks about a specifi'
- step 3 [final]  — thought: I now know the final answer

*Auto notes:* gold paper never retrieved

*ANNOTATION (fill in by hand):* where did the agent decide differently, and was it better or worse?

### qid 195  (category 4)
**Question:** What laser technology is used to perform corneal trephination for penetrating keratoplasty with orientation teeth and notches?

**Gold:** Femtosecond laser

**Fixed chain:** pred = `Femtosecond laser`  (EM 1), gold paper rank 1

**Agent:** pred = `Femtosecond laser technology`  (EM 0); stop = final; steps = 4; tools = {'query_reformulator': 1, 'generate_reasoning_query': 1, 'retrieve_from_index': 1}; gold paper seen = True

- step 1 [action] `query_reformulator(laser technology corneal trephination penetrating keratoplasty orientation teeth)`  — thought: I should first reformulate the question into a keyword-style search query.  → obs: 'Search query: laser technology corneal trephination penetrating keratoplasty orientation t'
- step 2 [action] `generate_reasoning_query(laser technology corneal trephination penetrating keratoplasty orientation teeth)`  — thought: I should generate a reasoning query to guide the search.  → obs: 'Reasoning (use it as the query for retrieve_from_index): The question involves laser techn'
- step 3 [action] `retrieve_from_index(The reasoning query for laser technology corneal trephination penetrating kerato)`  — thought: I should now search PubMed using the generated reasoning query.  → obs: '[1] Orientation teeth in nonmechanical femtosecond laser corneal trephination for penetrat'
- step 4 [final]  — thought: I now know the final answer

*Auto notes:* none

*ANNOTATION (fill in by hand):* where did the agent decide differently, and was it better or worse?
