## Project 2

For this project you will be applying supervised learning to solve a problem for an organisation
You will use your imagination to create your own project brief and then execute your plan to deliver real results
This project could be part of your final portfolio but don't sweat it, you can always improve it after your final presentation.

During this project you will notice that concepts that you thought you understood are more difficult to apply than you thought.
This is normal. Review your notes, think hard and reach out for help whenever you need.

You will be evaluated on your process not on the performance of your model
Some problems are just really difficult to crack and don't give you great results.
If you master the process you will get great results when the other factors are in place.

When you pass this project, you can consider yourself a Data Scientist


#####################
## Stage 1 - Brief ##
#####################

Milestone 1 - Project Brief/Planning - On 2021-02-16

Describe :
-What problem you would like to solve,
- who would benefit from it (AKA, identify your stakeholder/s),
- what data will you use and what do you think the target and predictors will be.
- how a machine learning model would contribute to the solution
- how will you measure success?

You will present your brief at the beginning of our session on the 16th

Guidelines:
- Problem and stakeholder selection:
  - Commercial problems make it easier for prospective employers to relate to your project. Avoid research questions. Show them the money!
  - Organisational stakeholders (someone working in a certain department in an organisation) are easier to focus on. They have KPIs how will you help them to improve them?
  - These organisations/stakeholders don't need to exist in real life. It just needs to be plausible that they exist.
  - Your stakeholder could also be a certain type of customer or citizen. In that case you would play the role of a start up / social enterprise. Imagine your KPIs and make a clear connection between the problem
  - The reason why we think of a stakeholder is to focus our effort in solving problems, not just building funcy, high performance models.
- Dataset selection:
  - The more unique your data the better. Avoid using datasets from kaggle or typical examples used for demonstrating concepts for your project.
  - This project is about modelling not data collection. Feel free to use the data that you collected already or some other that you can get easily.
  - If you don't have your hands on the data by the time you hand in this brief, give yourself a maximum of 1 day to get any data to work with and move on.
- Machine learning connection:
  - Explain how the model that you will build will help to solve the problem for your stakeholder.
  - Describe any additional steps that would need to be taken after the model is trained and selected in order to solve the problem.
  - Make a clear connection between the stakeholder KPIs and the 

*KPIs: Key Performance Indicators (very specific targets people in organisations are evaluated on)

#########################
## Stage 2 - Execution ##
#########################

The DUE DATES for this stage are the absolute latest you should be completing them
You should aim to complete this project in a shorter period of time
Things you can't anticipate will delay you. Take that into account when planning.


Milestone 2 - Baseline Model trained - Due by 2021-02-18
- Clean your data and do some Exploratory Data Analysis
  - Remember these two improve each other and may have to go through multiple iterations
  - Split your data in 2 or 3 buckets (depending on whether you would be using cross-validation or not respectively)
  - Different learners require different types of data preparation. You may have to revisit this as you go through Milestone 3
- Come up with a general idea of what you are expecting to obtain from your analysis
- Choose the learner that you think would perform best given what you know about your data and the problem
- Fit the simplest version possible

Milestone 3 - Final Model selected - Due by 2021-02-22
- Decide the evaluation metric that you will use to compare the different models
- Think what variations could be worth exploring:
  - Different learners
  - Different hyperparameters
  - Different optimisation techniques
- You will not have time to try them all. Prioritise them. Then run them.
- Keep a record of all your trials and the resulting performance.
- If you are solving a classification problem remember to choose your threshold and justify your decision (remember the formula).
- Compare the performance across your records and choose your final model
- Only then check your Test dataset

Milestone 4 - Model interpretation completed - Due by 2021-02-23
- How important is each of your predictors
- How does your prediction change when you increase/decrease each of your predictors (in their original units) by a certain amount?
- In which way do you see this information confirming/challenge what you expected?
- How would you explain this to a non-technical person who needs to approve your project?

- How good is your model?
- How frequent/how big will be the error?
- What does this mean for the problem that you are trying to solve?

Milestone 5 - Problem solution - Due by 2021-02-24
- It could be as simple as explaining the rules that your stakeholder should follow based on your model
- It could be as complicated as creating software for decision making
- Whatever you go for, make sure you describe how your stakeholder will interact with your model (if at all)

############################
## Stage 3 - Presentation ##
############################

Milestone 6 - Project presentation - On 2021-02-25
- Your presentation should last between 5 and 15 minutes
- Prepare some slides and rehearse a little bit. Allocate time for this.
- It should cover your work on the previous 5 milestones
- Assume your audience has no technical knowledge for the parts of the presentation referring to Milestones 1 and 5
- After your presentation you could be asked questions about:
  - The context of your project
  - Technical concepts to check your understanding
  - Your technical process
  - The impact of your project (on your fictional or real stakeholder)
- Provide us with a link to a clean github repository containing the code and data
  - It should contain a readme file explaining your project (you can build it with what you used for your presentation or viceversa)
  - The repo should have an easy to navigate structure. It it clear which notebook you should run?
  - Any depricated code has either been removed taken to another folder
  - All Jupyter notebook/s cells are run and show their output (which should be trimmed if too long)
  - No error or warning messages are shown unless is part of your narrative. (but don't supress errors and warnings artificially)




