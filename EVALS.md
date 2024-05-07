Work In Progress
===
A collection of failed attempts and notes.

## Example outout of a run on a random Llama-3-8B fp16 via LMStudio API
```
lm_eval \
    --model gguf \
    --model_args pretrained="$MODEL_NAME",base_url="$MODEL_NAME" \
    --tasks gsm8k \
    --batch_size 1

Total Time: 1:48:13 @ ~4.92s/it

|Tasks|Version|     Filter     |n-shot|  Metric   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|-----:|---|-----:|
|gsm8k|      3|strict-match    |     5|exact_match|0.0099|±  |0.0027|
|     |       |flexible-extract|     5|exact_match|0.0402|±  |0.0054|
```

## Borked Testing
```bash
git clone git@github.com:Vahe1994/AQLM.git
cd AQLM
pip install -r requirements.txt
pip install -r lm-evaluation-harness/requirements.txt
export CUDA_VISIBLE_DEVICES=0
export QUANTZED_MODEL="/root/.cache/huggingface/hub/models--ISTA-DASLab--Meta-Llama-3-8B-Instruct-AQLM-2Bit-1x16/snapshots/eb41f2ac68972c272f7de2a92c052ff339af34b4/"
export MODEL_PATH="ISTA-DASLab/Meta-Llama-3-8B-Instruct-AQLM-2Bit-1x16"
#export DATASET=<INSERT DATASET NAME OR PATH TO CUSTOM DATA>
#export WANDB_PROJECT=MY_AQ_LM_EVAL
#export WANDB_NAME=COOL_EVAL_NAME

python lmeval.py \
    --model hf-causal \
    --model_args pretrained=$MODEL_PATH,dtype=float16,use_accelerate=True \
    --load $QUANTZED_MODEL \
    --tasks gsm8k \
    --batch_size 1

#    --tasks winogrande,piqa,hellaswag,arc_easy,arc_challenge \
# install evaluation harness
#git clone https://github.com/EleutherAI/lm-evaluation-harness
#cd lm-evaluation-harness
#pip install -e .
#cd ..
##pip install aqlm[gpu]
#
##pip install lm_eval[vllm]
##--model_args pretrained="$MODEL_NAME",tensor_parallel_size={GPUs_per_model},dtype=auto,gpu_memory_utilization=0.8,data_parallel_size={model_replicas} \
export MODEL_NAME="ISTA-DASLab/Meta-Llama-3-8B-Instruct-AQLM-2Bit-1x16"
lm_eval \
    --model vllm \
    --model_args pretrained="$MODEL_NAME",dtype=auto,gpu_memory_utilization=0.8 \
    --tasks mmlu \
    --batch_size auto
#
## run
#export MODEL_NAME="ISTA-DASLab/Meta-Llama-3-8B-Instruct-AQLM-2Bit-1x16"
#lm_eval \
#    --model hf \
#    --model_args pretrained="$MODEL_NAME" \
#    --tasks winogrande,piqa,hellaswag,arc_easy,arc_challenge,gsm8k,mmlu \
#    --device cuda:0 \
#    --batch_size 1
#
    --model_args model="$LLM_MODEL",base_url="$API_URL"

('anagrams1', 'anagrams2', 'anli_r1', 'anli_r2', 'anli_r3',
'arc_challenge', 'arc_easy', 'arithmetic_1dc', 'arithmetic_2da',
'arithmetic_2dm', 'arithmetic_2ds', 'arithmetic_3da',
'arithmetic_3ds', 'arithmetic_4da', 'arithmetic_4ds',
'arithmetic_5da', 'arithmetic_5ds', 'blimp_adjunct_island',
'blimp_anaphor_gender_agreement', 'blimp_anaphor_number_agreement',
'blimp_animate_subject_passive', 'blimp_animate_subject_trans',
'blimp_causative', 'blimp_complex_NP_island',
'blimp_coordinate_structure_constraint_complex_left_branch',
'blimp_coordinate_structure_constraint_object_extraction',
'blimp_determiner_noun_agreement_1', 'blimp_determiner_noun_agreement_2',
'blimp_determiner_noun_agreement_irregular_1',
'blimp_determiner_noun_agreement_irregular_2',
'blimp_determiner_noun_agreement_with_adj_2',
'blimp_determiner_noun_agreement_with_adj_irregular_1',
'blimp_determiner_noun_agreement_with_adj_irregular_2',
'blimp_determiner_noun_agreement_with_adjective_1',
'blimp_distractor_agreement_relational_noun',
'blimp_distractor_agreement_relative_clause', 'blimp_drop_argument',
'blimp_ellipsis_n_bar_1', 'blimp_ellipsis_n_bar_2',
'blimp_existential_there_object_raising',
'blimp_existential_there_quantifiers_1',
'blimp_existential_there_quantifiers_2',
'blimp_existential_there_subject_raising',
'blimp_expletive_it_object_raising', 'blimp_inchoative',
'blimp_intransitive', 'blimp_irregular_past_participle_adjectives',
'blimp_irregular_past_participle_verbs',
'blimp_irregular_plural_subject_verb_agreement_1',
'blimp_irregular_plural_subject_verb_agreement_2',
'blimp_left_branch_island_echo_question',
'blimp_left_branch_island_simple_question',
'blimp_matrix_question_npi_licensor_present', 'blimp_npi_present_1',
'blimp_npi_present_2', 'blimp_only_npi_licensor_present',
'blimp_only_npi_scope', 'blimp_passive_1',
'blimp_passive_2', 'blimp_principle_A_c_command',
'blimp_principle_A_case_1', 'blimp_principle_A_case_2',
'blimp_principle_A_domain_1', 'blimp_principle_A_domain_2',
'blimp_principle_A_domain_3', 'blimp_principle_A_reconstruction',
'blimp_regular_plural_subject_verb_agreement_1',
'blimp_regular_plural_subject_verb_agreement_2',
'blimp_sentential_negation_npi_licensor_present',
'blimp_sentential_negation_npi_scope',
'blimp_sentential_subject_island', 'blimp_superlative_quantifiers_1',
'blimp_superlative_quantifiers_2', 'blimp_tough_vs_raising_1',
'blimp_tough_vs_raising_2', 'blimp_transitive', 'blimp_wh_island',
'blimp_wh_questions_object_gap', 'blimp_wh_questions_subject_gap',
'blimp_wh_questions_subject_gap_long_distance', 'blimp_wh_vs_that_no_gap',
'blimp_wh_vs_that_no_gap_long_distance', 'blimp_wh_vs_that_with_gap',
'blimp_wh_vs_that_with_gap_long_distance', 'boolq', 'cb', 'cola',
'copa', 'coqa', 'crows_pairs_english', 'crows_pairs_english_age',
'crows_pairs_english_autre', 'crows_pairs_english_disability',
'crows_pairs_english_gender', 'crows_pairs_english_nationality',
'crows_pairs_english_physical_appearance',
'crows_pairs_english_race_color', 'crows_pairs_english_religion',
'crows_pairs_english_sexual_orientation',
'crows_pairs_english_socioeconomic',
'crows_pairs_french', 'crows_pairs_french_age',
'crows_pairs_french_autre', 'crows_pairs_french_disability',
'crows_pairs_french_gender', 'crows_pairs_french_nationality',
'crows_pairs_french_physical_appearance', 'crows_pairs_french_race_color',
'crows_pairs_french_religion', 'crows_pairs_french_sexual_orientation',
'crows_pairs_french_socioeconomic', 'cycle_letters', 'drop', 'ethics_cm',
'ethics_deontology', 'ethics_justice', 'ethics_utilitarianism',
'ethics_utilitarianism_original', 'ethics_virtue', 'gsm8k', 'headqa',
'headqa_en', 'headqa_es', 'hellaswag', 'hendrycksTest-abstract_algebra',
'hendrycksTest-anatomy', 'hendrycksTest-astronomy',
'hendrycksTest-business_ethics', 'hendrycksTest-clinical_knowledge',
'hendrycksTest-college_biology', 'hendrycksTest-college_chemistry',
'hendrycksTest-college_computer_science',
'hendrycksTest-college_mathematics',
'hendrycksTest-college_medicine', 'hendrycksTest-college_physics',
'hendrycksTest-computer_security', 'hendrycksTest-conceptual_physics',
'hendrycksTest-econometrics', 'hendrycksTest-electrical_engineering',
'hendrycksTest-elementary_mathematics', 'hendrycksTest-formal_logic',
'hendrycksTest-global_facts', 'hendrycksTest-high_school_biology',
'hendrycksTest-high_school_chemistry',
'hendrycksTest-high_school_computer_science',
'hendrycksTest-high_school_european_history',
'hendrycksTest-high_school_geography',
'hendrycksTest-high_school_government_and_politics',
'hendrycksTest-high_school_macroeconomics',
'hendrycksTest-high_school_mathematics',
'hendrycksTest-high_school_microeconomics',
'hendrycksTest-high_school_physics',
'hendrycksTest-high_school_psychology',
'hendrycksTest-high_school_statistics',
'hendrycksTest-high_school_us_history',
'hendrycksTest-high_school_world_history', 'hendrycksTest-human_aging',
'hendrycksTest-human_sexuality', 'hendrycksTest-international_law',
'hendrycksTest-jurisprudence', 'hendrycksTest-logical_fallacies',
'hendrycksTest-machine_learning', 'hendrycksTest-management',
'hendrycksTest-marketing', 'hendrycksTest-medical_genetics',
'hendrycksTest-miscellaneous', 'hendrycksTest-moral_disputes',
'hendrycksTest-moral_scenarios', 'hendrycksTest-nutrition',
'hendrycksTest-philosophy', 'hendrycksTest-prehistory',
'hendrycksTest-professional_accounting',
'hendrycksTest-professional_law', 'hendrycksTest-professional_medicine',
'hendrycksTest-professional_psychology', 'hendrycksTest-public_relations',
'hendrycksTest-security_studies', 'hendrycksTest-sociology',
'hendrycksTest-us_foreign_policy', 'hendrycksTest-virology',
'hendrycksTest-world_religions', 'iwslt17-ar-en', 'iwslt17-en-ar',
'lambada_openai', 'lambada_openai_cloze', 'lambada_openai_mt_de',
'lambada_openai_mt_en', 'lambada_openai_mt_es', 'lambada_openai_mt_fr',
'lambada_openai_mt_it', 'lambada_standard', 'lambada_standard_cloze',
'logiqa', 'math_algebra', 'math_asdiv', 'math_counting_and_prob',
'math_geometry', 'math_intermediate_algebra', 'math_num_theory',
'math_prealgebra', 'math_precalc', 'mathqa', 'mc_taco', 'mnli',
'mnli_mismatched', 'mrpc', 'multirc', 'mutual', 'mutual_plus',
'openbookqa', 'pile_arxiv', 'pile_bookcorpus2', 'pile_books3',
'pile_dm-mathematics', 'pile_enron', 'pile_europarl', 'pile_freelaw',
'pile_github', 'pile_gutenberg', 'pile_hackernews', 'pile_nih-exporter',
'pile_opensubtitles', 'pile_openwebtext2', 'pile_philpapers',
'pile_pile-cc', 'pile_pubmed-abstracts', 'pile_pubmed-central',
'pile_stackexchange', 'pile_ubuntu-irc', 'pile_uspto', 'pile_wikipedia',
'pile_youtubesubtitles', 'piqa', 'prost', 'pubmedqa', 'qa4mre_2011',
'qa4mre_2012', 'qa4mre_2013', 'qasper', 'qnli', 'qqp', 'race',
'random_insertion', 'record', 'reversed_words', 'rte', 'sciq', 'squad2',
'sst', 'swag', 'toxigen', 'triviaqa', 'truthfulqa_gen', 'truthfulqa_mc',
'webqs', 'wic', 'wikitext', 'winogrande', 'wmt14-en-fr', 'wmt14-fr-en',
'wmt16-de-en', 'wmt16-en-de', 'wmt16-en-ro', 'wmt16-ro-en', 'wmt20-cs-en',
'wmt20-de-en', 'wmt20-de-fr', 'wmt20-en-cs', 'wmt20-en-de', 'wmt20-en-iu',
'wmt20-en-ja', 'wmt20-en-km', 'wmt20-en-pl', 'wmt20-en-ps', 'wmt20-en-ru',
'wmt20-en-ta', 'wmt20-en-zh', 'wmt20-fr-de', 'wmt20-iu-en', 'wmt20-ja-en',
'wmt20-km-en', 'wmt20-pl-en', 'wmt20-ps-en', 'wmt20-ru-en', 'wmt20-ta-en',
'wmt20-zh-en', 'wnli', 'wsc', 'wsc273')
```

## References
* [Evals Explained](https://medium.com/@ingridwickstevens/more-llm-acronyms-an-explainer-on-llama-3s-performance-benchmark-values-36722c6dcabb)
