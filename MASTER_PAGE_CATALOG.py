"""Master Page Type Catalog for LIC Form 300 (28-page version)
Generated from PaddleOCR scans of P02 and P05.

Page  Type                    Section                                  Key Fields
----  ----------------------  ---------------------------------------  --------------------------
  1   cover_page              Cover / Office Use                       photo, inward_no, proposal_no, receipt_date, deposit_amount
  2   proposer_details        Section-I: Personal Details              customer_id, ckye_no, name(first/mid/last), father_name, mother_name, gender, marital, spouse, dob, age, place_of_birth, nationality, citizenship, address, pincode
  3   kyc_occupation          Section-I: KYC/Occupation                residential_status, pan, aadhaar, gst, education, occupation, income, employer, duties, service_length
  4   medical_history         Section-I: Medical History               medical_consultation, hospital_admission, medical_examination_date, medical_category
  5   existing_policies       Section-I: Existing Policies             policy_number, insurer_name, plan_term, sum_assured, premium
  6   nominee_details         Section-I: Nominee & Appointee           nominee_name, nominee_relationship, nominee_age, nominee_address, appointee_name
  7   plan_details            Section-II: Proposed Plan Details         plan_name, plan_term, sum_assured, premium_mode, objective_of_insurance
  8   occupation_risk         Section-II: Occupation Risk              police_duty, hazardous_occupation
  9   plan_options            Section-II: Plan Options/Settlement       settlement_option, plan_specific_options
 10   health_habits           Section-III: Health/Habits Start          height_cm, weight_kg, disease_checkboxes_part1
 11   health_diseases         Section-III: Disease Declarations         disease_checkboxes_part2 (cancer, respiratory, etc.)
 12   health_family           Section-III: Family Health               usual_health_state, family_details, hereditary_conditions
 13   health_gynec            Section-III: Female Health / Spouse       gynecological_history, husband_details
 14   declaration_main        Section-IV: Main Declaration             proposer_declaration, signature
 15   declaration_declarant    Section-IV: Declarant Details            declarant_name, declarant_signature, declarant_address
 16   agent_details           Agent / Development Officer              agent_code, agent_name, branch_code, branch_name
 17   medical_examiner         Medical Examiner Section                 examiner_name, examiner_reg_no, examination_details
 18   settlement_addendum      Addendum: Settlement Option              settlement_option_details, maturity_benefit
 19   plan_specific_addendum   Addendum: Plan Specific (Dhan Sanchay 865) benefit_option, premium_paying_term
 20   plan_specific_addendum2  Addendum: Plan Specific (Jeevan Kiran 870, Jeevan Utsav 871) smoker_category, option_selection
 21   supplementary_risk       Supplementary Risk / Extra Premium       extra_premium_details
 22   supplementary_health     Supplementary Health / Family Details    family_health_history
 23   supplementary_la_details Supplementary LA Health Details          la_height, la_weight, la_illness
 24   annexure_1               Annexure-I: Previous Insurance           previous_insurance_details
 25   previous_policies        Previous Insurance Details               previous_policy_list
 26   suitability_start        Suitability Analysis Start               object_of_insurance, risk_cover_type
 27   suitability_continue     Suitability Analysis Continue            insurance_vs_income, pension/annuity, children
 28   suitability_last         Suitability Analysis End                 preferred_plan, preferred_term, preferred_sum_assured, preferred_mode, preferred_premium

KEY INSIGHT: Pages 1-7 + 10 + 16 + 28 contain ALL 43 target fields.
Pages 8-9, 11-15, 17-27 are supporting/medical/declaration pages with
minimal target fields (mostly checkboxes and supplementary info).
"""
