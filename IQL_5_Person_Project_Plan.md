# Reinforcement Learning Project Plan — Implicit Q-Learning (IQL)

## 1. Project Overview

**Course:** Reinforcement Learning  
**Algorithm:** Implicit Q-Learning (IQL)  
**Team size:** 5 people  
**Suggested environment:** HalfCheetah-v5  
**Suggested offline dataset:** `mujoco/halfcheetah/medium-v0`  
**Suggested stack:** Python, PyTorch, Gymnasium, Minari, Jupyter Notebook / Google Colab

### Main project goal

The goal of the project is to demonstrate how **Implicit Q-Learning (IQL)** can learn a strong policy from an offline reinforcement learning dataset and then use this policy to perform well in a compatible environment.

The final project should contain:

- a working IQL implementation or adapted existing implementation;
- a compatible RL environment;
- offline training data;
- training and evaluation;
- a demo showing the trained agent performing well;
- a Jupyter Notebook / Google Colab notebook explaining the algorithm;
- a simple manual example showing how IQL updates its value estimates and/or policy;
- final presentation with results.

---

# 2. High-Level Pipeline

```text
Offline Dataset
      |
      |  (state, action, reward, next_state, done)
      v
+----------------------+
| Implicit Q-Learning  |
+----------------------+
      |
      | learns:
      |
      +--> Value Function V(s)
      |
      +--> Q-Function Q(s, a)
      |
      +--> Policy pi(a | s)
      |
      v
Trained Agent
      |
      v
HalfCheetah-v5
      |
      v
Evaluation + Demo
```

---

# 3. Team Structure

| Person | Role | Main Responsibility |
|---|---|---|
| Person 1 | Team Lead / Integration | Project structure, repository, integration, final Colab |
| Person 2 | Environment & Dataset | Environment, dataset loading, preprocessing |
| Person 3 | IQL Implementation | Networks, losses, training updates |
| Person 4 | Experiments & Evaluation | Metrics, evaluation, plots, demo/video |
| Person 5 | Theory / Notebook / Presentation | Explanation, manual example, slides, final narrative |

The key idea is to avoid having all five people independently modify the same part of the algorithm. One person should own the integration, while the others own clearly separated components.

---

# 4. Person 1 — Team Lead / Integration

## Main responsibility

Person 1 is responsible for keeping the whole project consistent and runnable.

This person should make sure that all components created by other team members work together correctly.

## Tasks

- Create and maintain the GitHub repository.
- Define the project structure.
- Create shared configuration files.
- Define dependency versions.
- Merge code from other members.
- Connect the dataset loader with the IQL implementation.
- Connect the trained policy with the evaluation environment.
- Build the final Colab / Jupyter Notebook.
- Perform the final end-to-end test.
- Make sure the project runs from a clean environment.

## Suggested repository structure

```text
iql-project/
|
├── notebooks/
│   └── final_iql_project.ipynb
|
├── src/
│   ├── dataset.py
│   ├── environment.py
│   ├── networks.py
│   ├── iql.py
│   ├── train.py
│   └── evaluate.py
|
├── results/
│   ├── metrics.csv
│   └── plots/
|
├── videos/
│   └── iql_halfcheetah_demo.mp4
|
├── checkpoints/
│   └── iql_halfcheetah.pt
|
├── requirements.txt
├── README.md
└── PROJECT_PLAN.md
```

## Final deliverable

Person 1 must deliver:

- working repository;
- fully integrated code;
- final notebook;
- reproducible project;
- successful `Run all` execution in a clean Colab/Jupyter session.

## Definition of done

```text
[ ] Repository exists
[ ] All dependencies are documented
[ ] Dataset loader works
[ ] IQL training works
[ ] Evaluation works
[ ] Demo works
[ ] Final notebook runs from top to bottom
```

---

# 5. Person 2 — Environment & Offline Dataset

## Main responsibility

Person 2 is responsible for understanding the environment and preparing the offline dataset used by IQL.

## Suggested environment

**HalfCheetah-v5**

The person should explain:

- what the agent observes;
- what actions the agent can take;
- what reward means;
- what the objective is;
- why this environment is compatible with IQL.

## Environment explanation

Example structure:

```text
Environment: HalfCheetah-v5

Observation:
Continuous state vector describing the robot configuration and velocity.

Action:
Continuous vector controlling the robot joints.

Goal:
Move forward as efficiently as possible.

Reward:
Mainly rewards forward movement and penalizes excessive control effort.
```

## Dataset

Suggested dataset:

```text
mujoco/halfcheetah/medium-v0
```

The dataset should provide transitions:

```text
(state, action, reward, next_state, done)
```

## Tasks

- Install and configure Gymnasium / Minari.
- Load the offline dataset.
- Inspect the dataset.
- Check observation and action dimensions.
- Convert the dataset to a format convenient for PyTorch.
- Create batch sampling logic.
- Check for invalid or missing values.
- Document dataset statistics.

## Suggested batch format

```python
batch = {
    "states": ...,
    "actions": ...,
    "rewards": ...,
    "next_states": ...,
    "dones": ...
}
```

## Dataset analysis

At minimum collect:

```text
Number of episodes
Number of transitions
Observation dimension
Action dimension
Reward statistics
Episode return statistics
```

Optional visualizations:

- reward distribution;
- episode return distribution;
- episode length distribution.

## Final deliverable

Person 2 must deliver:

- environment setup;
- dataset loader;
- preprocessing logic;
- batch sampling;
- short environment description;
- dataset statistics.

## Definition of done

```text
[ ] Environment runs
[ ] Dataset loads
[ ] Observation dimension is known
[ ] Action dimension is known
[ ] Batch sampling works
[ ] Dataset statistics are available
[ ] Person 3 can directly use the batches
```

---

# 6. Person 3 — IQL Implementation

## Main responsibility

Person 3 owns the core algorithm.

This is the most technically demanding role.

The person may implement IQL from scratch or adapt an existing implementation, but they must understand the algorithm well enough to explain every major update.

## Core components

The implementation should contain:

```text
Value network V(s)

Q-network Q1(s, a)

Q-network Q2(s, a)

Policy pi(a | s)

Target networks if required
```

---

## 6.1 Value Function Update

IQL learns a value function using expectile regression.

Define:

\[
u = Q(s,a) - V(s)
\]

Then optimize:

\[
L_V =
|\tau - \mathbb{1}(u < 0)|u^2
\]

The parameter \(\tau\) controls which part of the Q-value distribution the value function focuses on.

---

## 6.2 Q-Function Update

For a transition:

\[
(s_t, a_t, r_t, s_{t+1})
\]

the target is:

\[
y = r_t + \gamma V(s_{t+1})
\]

Then the Q-loss is:

\[
L_Q =
(Q(s_t,a_t) - y)^2
\]

With two Q-functions, both are updated toward the same target.

---

## 6.3 Policy Update

First compute the advantage:

\[
A(s,a) = Q(s,a) - V(s)
\]

Then calculate an exponential weight:

\[
w = \exp(\beta A(s,a))
\]

The policy is trained to imitate dataset actions, but actions with larger positive advantage receive more weight.

Conceptually:

```text
Good actions in the dataset
        ->
larger weight
        ->
policy imitates them more strongly

Bad actions in the dataset
        ->
smaller weight
        ->
policy imitates them less
```

---

## 6.4 Training Function

A simplified interface could look like:

```python
def update(batch):
    value_loss = update_value(batch)
    q_loss = update_q(batch)
    policy_loss = update_policy(batch)

    return {
        "value_loss": value_loss,
        "q_loss": q_loss,
        "policy_loss": policy_loss,
    }
```

## Tasks

- Implement/adapt all networks.
- Implement the expectile loss.
- Implement Q update.
- Implement policy update.
- Add optimizers.
- Add model saving/loading.
- Return training metrics.
- Document all important hyperparameters.

## Important hyperparameters

At minimum:

```text
gamma
tau
beta
batch_size
learning_rate
number_of_training_steps
hidden_layer_sizes
```

## Final deliverable

Person 3 must deliver:

- working IQL model;
- training update function;
- all loss functions;
- checkpoint save/load functionality;
- documentation of important hyperparameters.

## Definition of done

```text
[ ] Networks initialize correctly
[ ] Batch passes through the model
[ ] Value loss works
[ ] Q loss works
[ ] Policy loss works
[ ] One training step works
[ ] Losses are finite
[ ] Checkpoint can be saved
[ ] Checkpoint can be loaded
```

---

# 7. Person 4 — Experiments, Evaluation & Demo

## Main responsibility

Person 4 answers the main practical question:

> Does the trained IQL agent actually perform well in the environment?

## Tasks

- Run evaluation episodes.
- Measure returns.
- Track training metrics.
- Create plots.
- Compare early and final performance.
- Record the demo.
- Save evaluation results.

## Minimum evaluation metrics

```text
Episode return
Average return
Standard deviation
Number of evaluation episodes
```

Example:

```text
Evaluation over 10 episodes

Episode 1: ...
Episode 2: ...
Episode 3: ...
...

Mean return: ...
Standard deviation: ...
```

## Recommended plots

At minimum:

```text
Value loss
Q loss
Policy loss
Evaluation return
```

Do not create unnecessary plots just to increase quantity.

The plots should clearly help explain whether training is stable and whether the agent improves.

---

## Demo

The final presentation must contain a demo of the agent.

A good demo can contain:

```text
Random / untrained policy
        vs
Trained IQL policy
```

Possible result:

```text
Before training:
The HalfCheetah moves poorly or fails to move forward.

After training:
The HalfCheetah consistently moves forward.
```

Save the result as something like:

```text
videos/iql_halfcheetah_demo.mp4
```

## Final deliverable

Person 4 must deliver:

- evaluation script;
- metrics;
- plots;
- final evaluation table;
- demo video;
- short interpretation of results.

## Definition of done

```text
[ ] Evaluation environment works
[ ] Trained checkpoint loads
[ ] Evaluation over multiple episodes works
[ ] Mean return is calculated
[ ] Standard deviation is calculated
[ ] Plots are generated
[ ] Demo video is recorded
```

---

# 8. Person 5 — Theory, Manual Example, Notebook Narrative & Presentation

## Main responsibility

Person 5 is responsible for making the project understandable.

The code may work perfectly, but the team must still be able to explain:

- what problem IQL solves;
- why offline RL is different;
- how IQL learns;
- what the main update equations mean;
- why the agent improves.

---

## 8.1 Explain Offline RL

A simple explanation:

```text
In online reinforcement learning, the agent can continuously interact with the environment
and collect new experience.

In offline reinforcement learning, the agent only receives a fixed dataset.

It cannot freely explore the environment during training.
```

The dataset contains:

\[
(s,a,r,s')
\]

transitions collected beforehand.

---

## 8.2 Explain the main problem

A standard Q-learning algorithm may assign unrealistically high Q-values to actions that are not well represented in the offline dataset.

IQL tries to avoid relying on unseen actions during offline training.

---

## 8.3 Manual Numerical Example

This is especially important because the mentor explicitly requires a small example demonstrating how the algorithm updates values or policy.

### Example

Assume:

\[
Q(s,a)=8
\]

and

\[
V(s)=5
\]

Then:

\[
A(s,a)=Q(s,a)-V(s)
\]

\[
A(s,a)=8-5=3
\]

If:

\[
\beta=1
\]

then:

\[
w=e^{3}\approx20.09
\]

Interpretation:

```text
This action is much better than the value baseline.

Therefore, IQL gives this action a large weight
during the policy update.

The policy becomes more likely to reproduce similar good actions.
```

### Bad action example

Assume:

\[
Q(s,a)=3
\]

and:

\[
V(s)=5
\]

Then:

\[
A(s,a)=3-5=-2
\]

and:

\[
w=e^{-2}\approx0.135
\]

Interpretation:

```text
This action is worse than expected for this state.

It receives a small policy weight.

The policy will not strongly imitate it.
```

---

## Tasks

- Write the theoretical explanation.
- Create diagrams.
- Prepare the manual numerical example.
- Help structure the final notebook.
- Create proposal slides.
- Create final presentation slides.
- Make sure explanations match the actual implementation.

## Final deliverable

Person 5 must deliver:

- theoretical IQL explanation;
- offline RL explanation;
- manual update example;
- final presentation;
- explanatory Markdown sections in the notebook.

## Definition of done

```text
[ ] Offline RL is explained
[ ] IQL motivation is explained
[ ] V update is explained
[ ] Q update is explained
[ ] Policy update is explained
[ ] Numerical example is ready
[ ] Slides are complete
[ ] Slides match the actual implementation
```

---

# 9. Project Stages

# Stage 1 — Proposal

The proposal does not need the complete implementation yet.

The purpose is to explain:

- what environment is selected;
- what problem the agent solves;
- what IQL is;
- why IQL is compatible with this environment;
- how the project will be implemented;
- what the final demo will show.

## Responsibilities

### Person 1

- define overall architecture;
- verify project scope;
- review proposal.

### Person 2

Prepare:

```text
Environment
Observation space
Action space
Reward
Goal
Dataset
```

### Person 3

Prepare:

```text
What is IQL?
How does IQL work?
Main components of the algorithm
```

### Person 4

Prepare:

```text
Evaluation strategy
Expected metrics
Expected demo
```

### Person 5

- combine everything into slides;
- simplify explanations;
- prepare diagrams.

---

# 10. Suggested Proposal Presentation

A simple proposal can contain around 5–6 slides.

## Slide 1 — Project Title

```text
Implicit Q-Learning for Offline Reinforcement Learning
on HalfCheetah
```

Include:

- team members;
- algorithm;
- environment.

---

## Slide 2 — Environment

Explain:

- HalfCheetah-v5;
- observation;
- action;
- reward;
- goal.

---

## Slide 3 — Offline Reinforcement Learning

Explain:

```text
Fixed dataset
No online exploration during training
Agent learns only from previously collected experience
```

---

## Slide 4 — How IQL Works

Show:

```text
Dataset
  |
  v
Q(s,a)
  |
  v
V(s)
  |
  v
Advantage
  |
  v
Weighted Policy Learning
```

---

## Slide 5 — Project Pipeline

```text
Offline Dataset
      |
      v
IQL Training
      |
      v
Trained Policy
      |
      v
HalfCheetah
      |
      v
Evaluation
      |
      v
Demo
```

---

## Slide 6 — Expected Result

Explain that the final project will show:

- trained agent;
- evaluation metrics;
- learning curves;
- demo video;
- manual IQL update example.

---

# 11. Stage 2 — Minimal Working Version

The goal is not yet to train the perfect agent.

The goal is to make the whole pipeline work.

## Person 2

Must achieve:

```text
Dataset loads
        |
        v
Batch sampling works
```

## Person 3

Must achieve:

```text
Batch
 |
 v
IQL update
 |
 v
Loss values
```

## Person 1

Connects both components.

Expected minimal test:

```python
batch = sample_batch()

metrics = iql.update(batch)

print(metrics)
```

Expected output should contain something like:

```text
value_loss: ...
q_loss: ...
policy_loss: ...
```

---

# 12. Stage 3 — Full Training

After the minimal version works, start full training.

Pipeline:

```text
Offline Dataset
      |
      v
IQL
      |
      v
100k / 500k / 1M updates
      |
      v
Checkpoint
```

Save checkpoints periodically:

```text
checkpoints/
├── iql_100k.pt
├── iql_500k.pt
└── iql_final.pt
```

## Responsibilities

### Person 3

- training;
- algorithm debugging;
- hyperparameters.

### Person 4

- evaluation;
- monitoring;
- plots.

### Person 1

- integration;
- reproducibility.

---

# 13. Stage 4 — Results

Person 4 collects:

```text
Final average return
Standard deviation
Training losses
Evaluation return
Demo video
```

Person 5 transforms these into presentation-ready explanations.

Person 1 verifies that the numbers in the slides match the actual output.

---

# 14. Stage 5 — Final Colab / Jupyter Notebook

The final notebook should not look like a giant code dump.

It should tell the story of the project.

Recommended structure:

```text
1. Introduction

2. Environment

3. Offline Reinforcement Learning

4. Dataset

5. What is IQL?

6. IQL Architecture

7. Manual Numerical Example

8. Implementation

9. Training

10. Training Metrics

11. Evaluation

12. Demo

13. Results

14. Conclusion
```

---

# 15. Suggested Final Notebook Structure

## Section 1 — Introduction

Explain the goal of the project.

---

## Section 2 — Environment

Describe HalfCheetah.

---

## Section 3 — Offline RL

Explain why training uses a fixed dataset.

---

## Section 4 — Dataset

Show:

```text
Number of transitions
Observation dimension
Action dimension
Reward statistics
```

---

## Section 5 — IQL

Explain the three core parts:

```text
Value function
Q-function
Policy
```

---

## Section 6 — Manual Update Example

Use real numbers to explain:

```text
Q(s,a)
V(s)
Advantage
Policy weight
```

---

## Section 7 — Implementation

Show only the important parts of the code.

Avoid overwhelming the notebook with unnecessary implementation details.

---

## Section 8 — Training

Explain:

```text
Number of steps
Batch size
Learning rate
Gamma
Tau
Beta
```

---

## Section 9 — Metrics

Show training curves.

---

## Section 10 — Evaluation

Run the trained agent for multiple episodes.

---

## Section 11 — Demo

Render the environment and show the final agent.

---

## Section 12 — Conclusion

Explain:

- whether IQL worked;
- what performance was achieved;
- main challenges;
- possible improvements.

---

# 16. Final Presentation — Who Presents What

If all five people should speak, divide the presentation like this:

| Person | Presentation Part |
|---|---|
| Person 1 | Problem, goal, full project pipeline |
| Person 2 | Environment and dataset |
| Person 3 | IQL algorithm and updates |
| Person 4 | Training, evaluation, metrics, results |
| Person 5 | Manual numerical example, demo, conclusion |

This ensures that every team member has a meaningful technical contribution.

---

# 17. Individual Deliverables

Every member should have a concrete measurable result.

## Person 1

```text
Working integrated repository
Final Colab
Dependency management
Final end-to-end test
```

## Person 2

```text
Environment setup
Dataset loader
Batch sampler
Dataset statistics
```

## Person 3

```text
IQL model
Value update
Q update
Policy update
Training loop
Checkpoint handling
```

## Person 4

```text
Evaluation code
Metrics
Plots
Demo video
Final results
```

## Person 5

```text
Theory explanation
Manual example
Proposal slides
Final slides
Notebook explanations
```

---

# 18. Team Workflow

Recommended workflow:

```text
                 Person 1
              Integration Lead
                    |
        +-----------+-----------+
        |           |           |
        v           v           v
     Person 2    Person 3    Person 4
    Environment     IQL       Evaluation
        |           |           |
        +-----------+-----------+
                    |
                    v
                 Person 5
          Explanation + Slides
```

Person 5 should not wait until the final day.

They should continuously receive:

- environment description from Person 2;
- algorithm explanation from Person 3;
- plots and metrics from Person 4.

---

# 19. Recommended Git Workflow

Use branches:

```text
main

feature/environment-dataset

feature/iql

feature/evaluation

feature/notebook-presentation
```

Recommended process:

```text
Create branch
      |
      v
Implement task
      |
      v
Test locally
      |
      v
Pull Request
      |
      v
Person 1 reviews
      |
      v
Merge into main
```

Avoid everyone committing directly to `main`.

---

# 20. Project Milestones

## Milestone 1 — Proposal Ready

```text
[ ] Environment selected
[ ] Dataset selected
[ ] IQL understood
[ ] Pipeline diagram ready
[ ] Proposal slides ready
```

---

## Milestone 2 — Data Pipeline Ready

```text
[ ] Environment installs
[ ] Dataset downloads
[ ] Dataset loads
[ ] Batches can be sampled
```

---

## Milestone 3 — IQL Minimal Version

```text
[ ] Networks initialize
[ ] One update step works
[ ] Losses are finite
[ ] Model checkpoint saves
```

---

## Milestone 4 — Full Training

```text
[ ] Long training run works
[ ] Metrics are saved
[ ] Checkpoints are saved
```

---

## Milestone 5 — Evaluation

```text
[ ] Agent loads
[ ] Evaluation runs
[ ] Average return is calculated
[ ] Plots are generated
```

---

## Milestone 6 — Demo

```text
[ ] Environment renders
[ ] Trained policy performs correctly
[ ] Demo video is recorded
```

---

## Milestone 7 — Final Submission

```text
[ ] Colab runs from start to finish
[ ] Manual example is included
[ ] Results are included
[ ] Demo is included
[ ] Slides are ready
[ ] Every member knows their section
```

---

# 21. Main Risks

## Risk 1 — Environment does not run in Colab

Possible issues:

```text
MuJoCo installation
Rendering
Package version conflicts
```

Solution:

- test environment installation early;
- do not wait until the final week.

---

## Risk 2 — IQL training is unstable

Possible causes:

```text
Wrong loss implementation
Wrong normalization
Bad hyperparameters
Incorrect target calculation
```

Solution:

- first verify one training step;
- compare implementation with a trusted reference;
- log all losses.

---

## Risk 3 — Agent trains but demo fails

Possible causes:

```text
Different environment configuration
Action scaling mismatch
Wrong checkpoint
Rendering issues
```

Solution:

- start evaluation early;
- test saved checkpoints regularly.

---

## Risk 4 — Final notebook cannot reproduce results

Solution:

Person 1 must test:

```text
Factory reset runtime
        |
        v
Run all
```

before the presentation.

---

# 22. Recommended Priority Order

Do not start with presentation design.

Recommended order:

```text
1. Environment works

2. Dataset loads

3. One IQL update works

4. Full training works

5. Evaluation works

6. Demo works

7. Final notebook is cleaned

8. Final presentation is polished
```

---

# 23. Minimal Successful Project

If time becomes limited, the minimum acceptable version should still contain:

```text
Working HalfCheetah environment

Offline dataset

Working IQL training

Saved trained model

Evaluation over multiple episodes

Average return

Manual IQL update example

Demo video

Final Colab

Final presentation
```

Everything else is secondary.

---

# 24. Final Goal

By the end of the project, the team should be able to demonstrate this complete pipeline:

```text
Offline Dataset
      |
      v
Implicit Q-Learning
      |
      v
Learn V(s), Q(s,a), and Policy
      |
      v
Trained Policy
      |
      v
HalfCheetah Environment
      |
      v
Good Agent Performance
      |
      v
Evaluation + Demo + Explanation
```

The final result should not only show that the code works, but also demonstrate that the team understands **why IQL works, how it updates its value functions and policy, and how those updates lead to improved agent behavior**.
