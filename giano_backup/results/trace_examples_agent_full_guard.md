# Fixed chain (baseline_vanilla_rag_v3) vs dynamic agent (agent_full_guard)

Questions where EM differs: agent better 13, agent worse 24 (of 500).


## agent BETTER than the fixed chain

### qid 333  (category 1)
**Question:** Which gene is responsible for producing the catalytic region of aspartyl-tRNA synthetase in rat hepatic tissue?

**Gold:** DRS1

**Fixed chain:** pred = `DRS1 gene`  (EM 0), gold paper rank 1

**Agent:** pred = `DRS1`  (EM 1); stop = loop; steps = 4; tools = {'retrieve_from_index': 2}; gold paper seen = True

- step 1 [action] `retrieve_from_index(gene responsible for catalytic region of aspartyl-tRNA synthetase rat hepatic ti)`  — thought: I need to find the gene responsible for producing the catalytic region of aspartyl-tRNA synthetase in rat hepatic tissue  → obs: '[1] Molecular cloning and primary structure of cDNA encoding the catalytic domain of rat l'
- step 2 [action] `retrieve_from_index(DRS1 gene rat liver aspartyl-tRNA synthetase catalytic region)`  — thought: The question asks for the gene responsible for producing the catalytic region of aspartyl-tRNA synthetase in rat hepatic  → obs: '[1] Molecular cloning and primary structure of cDNA encoding the catalytic domain of rat l'
- step 3 [action] `retrieve_from_index(DRS1 gene rat liver aspartyl-tRNA synthetase catalytic region)`  — thought: The question asks for the gene responsible for producing the catalytic region of aspartyl-tRNA synthetase in rat hepatic  → obs: 'You already ran this exact action. Use a different input or tool, or give the Final Answer'
- step 4 [action] `retrieve_from_index(DRS1 gene rat liver aspartyl-tRNA synthetase catalytic region)`  — thought: The question asks for the gene responsible for producing the catalytic region of aspartyl-tRNA synthetase in rat hepatic  → obs: 'You already ran this exact action. Use a different input or tool, or give the Final Answer'
- step 5 [forced_stop] reason=loop

*Auto notes:* repeated an identical action; loop breaker stopped it

*ANNOTATION (fill in by hand):* where did the agent decide differently, and was it better or worse?

### qid 141  (category 4)
**Question:** What laboratory method involving enzymes is frequently employed to identify where small molecules with sequence specificity attach to DNA?

**Gold:** DNase I footprinting

**Fixed chain:** pred = `Restriction enzyme assay`  (EM 0), gold paper rank 2

**Agent:** pred = `DNase I footprinting`  (EM 1); stop = loop; steps = 4; tools = {'retrieve_from_index': 2}; gold paper seen = True

- step 1 [action] `retrieve_from_index(enzyme method DNA sequence specificity attachment identification)`  — thought: I need to find abstracts about laboratory methods involving enzymes for identifying small molecules with sequence specif  → obs: '[1] An integrated system for enzymatic cleavage and electrostretching of freely-suspended '
- step 2 [action] `retrieve_from_index(restriction enzyme DNA sequence specificity attachment identification)`  — thought: The abstracts discuss various methods for identifying small molecules with sequence specificity attaching to DNA, includ  → obs: '[1] An integrated system for enzymatic cleavage and electrostretching of freely-suspended '
- step 3 [action] `retrieve_from_index(restriction enzyme DNA sequence specificity attachment identification)`  — thought: The question asks for a laboratory method involving enzymes to identify small molecules with sequence specificity attach  → obs: 'You already ran this exact action. Use a different input or tool, or give the Final Answer'
- step 4 [error]  → obs: "Invalid format. Reply with 'Action:' and 'Action Input:', or with 'Final Answer:'."
- step 5 [forced_stop] reason=loop

*Auto notes:* repeated an identical action; loop breaker stopped it; format error

*ANNOTATION (fill in by hand):* where did the agent decide differently, and was it better or worse?

### qid 272  (category 4)
**Question:** Which assay is commonly used to assess opsonophagocytic activity in response to pneumococcal vaccination?

**Gold:** Opsonophagocytic activity (OPA) assay

**Fixed chain:** pred = `OPA`  (EM 0), gold paper rank 2

**Agent:** pred = `OPA assay`  (EM 1); stop = loop; steps = 3; tools = {'query_reformulator': 1, 'retrieve_from_index': 1}; gold paper seen = True

- step 1 [action] `query_reformulator(opsonophagocytic activity pneumococcal vaccination assay)`  — thought: I need to find information about assays used to assess opsonophagocytic activity after pneumococcal vaccination. I will   → obs: 'Search query: pneumococcal vaccination assay opsonophagocytic activity'
- step 2 [action] `query_reformulator(opsonophagocytic activity pneumococcal vaccination assay)`  — thought: I need to find information about assays used to assess opsonophagocytic activity after pneumococcal vaccination. I will   → obs: 'You already ran this exact action. Use a different input or tool, or give the Final Answer'
- step 3 [action] `query_reformulator(opsonophagocytic activity pneumococcal vaccination assay)`  — thought: I need to find information about assays used to assess opsonophagocytic activity after pneumococcal vaccination. I will   → obs: 'You already ran this exact action. Use a different input or tool, or give the Final Answer'
- step 4 [forced_stop] reason=loop
- step 4 [guardrail] `retrieve_from_index(pneumococcal vaccination assay opsonophagocytic activity)`  → obs: '[1] Immunogenicity and safety of 10-valent pneumococcal non-typeable Haemophilus influenza'

*Auto notes:* repeated an identical action; loop breaker stopped it

*ANNOTATION (fill in by hand):* where did the agent decide differently, and was it better or worse?


## agent WORSE than the fixed chain

### qid 53  (category 7)
**Question:** What is the name of the website that offers daily refreshed public health statistics and interactive data visualization features related to COVID-19 for each state and union territory in India?

**Gold:** covind19.org

**Fixed chain:** pred = `covind19.org`  (EM 1), gold paper rank 1

**Agent:** pred = `datahub.io`  (EM 0); stop = loop; steps = 4; tools = {'retrieve_from_index': 1, 'fetch_pubmed_live': 1}; gold paper seen = True

- step 1 [action] `retrieve_from_index(website provides daily refreshed public health statistics and interactive data v)`  — thought: I need to find the name of the website that provides daily refreshed public health statistics and interactive data visua  → obs: '[1] Comprehensive public health evaluation of lockdown as a non-pharmaceutical interventio'
- step 2 [action] `fetch_pubmed_live(COVID-19 statistics website India)`  — thought: The question asks for the name of a specific website related to COVID-19 statistics in India. The abstracts discuss COVI  → obs: '[1] Efficacy and safety of COVID-19 vaccines.\nBACKGROUND: Different forms of vaccines have'
- step 3 [action] `fetch_pubmed_live(COVID-19 statistics website India)`  — thought: The question asks for the name of a specific website related to COVID-19 statistics in India. The abstracts discuss COVI  → obs: 'You already ran this exact action. Use a different input or tool, or give the Final Answer'
- step 4 [action] `fetch_pubmed_live(COVID-19 statistics website India)`  — thought: The question asks for the name of a specific website related to COVID-19 statistics in India. The abstracts discuss COVI  → obs: 'You already ran this exact action. Use a different input or tool, or give the Final Answer'
- step 5 [forced_stop] reason=loop

*Auto notes:* repeated an identical action; loop breaker stopped it

*ANNOTATION (fill in by hand):* where did the agent decide differently, and was it better or worse?

### qid 78  (category 9)
**Question:** In which compartment of the plasma membrane is sphingomyelin resynthesized following tumour necrosis factor alpha-induced hydrolysis?

**Gold:** Cytosolic leaflet of the plasma membrane

**Fixed chain:** pred = `Cytosolic leaflet of the plasma membrane`  (EM 1), gold paper rank 1

**Agent:** pred = `Plasma membrane`  (EM 0); stop = loop; steps = 3; tools = {'retrieve_from_index': 1}; gold paper seen = True

- step 1 [action] `retrieve_from_index(sphingomyelin resynthesis tumour necrosis factor alpha hydrolysis plasma membran)`  — thought: I need to find the answer to this question. I will use the local corpus first.  → obs: '[1] The tumour necrosis factor-sensitive pool of sphingomyelin is resynthesized in a disti'
- step 2 [action] `retrieve_from_index(sphingomyelin resynthesis tumour necrosis factor alpha hydrolysis plasma membran)`  — thought: The question asks about the compartment of the plasma membrane where sphingomyelin is resynthesized after TNF alpha-indu  → obs: 'You already ran this exact action. Use a different input or tool, or give the Final Answer'
- step 3 [action] `retrieve_from_index(sphingomyelin resynthesis tumour necrosis factor alpha hydrolysis plasma membran)`  — thought: The question asks about the compartment of the plasma membrane where sphingomyelin is resynthesized following tumour nec  → obs: 'You already ran this exact action. Use a different input or tool, or give the Final Answer'
- step 4 [forced_stop] reason=loop

*Auto notes:* repeated an identical action; loop breaker stopped it

*ANNOTATION (fill in by hand):* where did the agent decide differently, and was it better or worse?

### qid 423  (category 2)
**Question:** What is the preferred treatment modality for symptomatic ossification of the posterior longitudinal ligament of the cervical spine?

**Gold:** Surgical decompression

**Fixed chain:** pred = `Surgical decompression`  (EM 1), gold paper rank 1

**Agent:** pred = `Anterior approach`  (EM 0); stop = loop; steps = 3; tools = {'query_reformulator': 1, 'retrieve_from_index': 1}; gold paper seen = True

- step 1 [action] `query_reformulator(preferred treatment modality for symptomatic ossification of the posterior longi)`  — thought: I need to find information about the preferred treatment for symptomatic ossification of the posterior longitudinal liga  → obs: 'Search query: preferred treatment modality for symptomatic ossification of the posterior l'
- step 2 [action] `query_reformulator(preferred treatment modality for symptomatic ossification of the posterior longi)`  — thought: I need to find the preferred treatment modality for symptomatic ossification of the posterior longitudinal ligament (OPL  → obs: 'You already ran this exact action. Use a different input or tool, or give the Final Answer'
- step 3 [action] `query_reformulator(preferred treatment modality for symptomatic ossification of the posterior longi)`  — thought: I need to find the preferred treatment modality for symptomatic ossification of the posterior longitudinal ligament (OPL  → obs: 'You already ran this exact action. Use a different input or tool, or give the Final Answer'
- step 4 [forced_stop] reason=loop
- step 4 [guardrail] `retrieve_from_index(preferred treatment modality for symptomatic ossification of the posterior longi)`  → obs: '[1] Surgical treatment for ossification of the posterior longitudinal ligament of the cerv'

*Auto notes:* repeated an identical action; loop breaker stopped it

*ANNOTATION (fill in by hand):* where did the agent decide differently, and was it better or worse?
