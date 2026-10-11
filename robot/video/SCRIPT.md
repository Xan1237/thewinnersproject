# datanode — four-minute narration

English; Vel/Tim alternate. No introductions. Real capture footage only.

## 000–031s · Vel · From robot decisions to training data

These robots are working through the same tasks with different AI planners. On the right, every move becomes a record: what the robot observed, what it chose, and what happened next. datanode turns successful attempts into training examples for a small local controller. This walkthrough follows that process from the room, through the decision log and training, to the gold robot running the learned policy.

Visual: Live overview, named planners and the gold local robot.

## 031–061s · Tim · One room, several controllers

The room contains a coffee table, a lectern, chairs, and obstacles. Each robot has the same movement controls. Its label identifies the planner guiding it: Astra, Fable, or Opus. The instruction can ask for more than one step, such as collecting a coffee and bringing it to the lectern. We record complete attempts, including blocked moves and failures, so the comparison reflects what actually happened.

Visual: Follow a frontier robot; room objects and collision responses.

## 061–092s · Vel · The planner and Jev

The planner writes short guidance from a description of the room and the recent actions. Jev then chooses a move, such as turning left, walking forward, or reaching for the cup. The panel shows both parts: the planner's instruction and Jev's choice with its confidence. The simulator applies the move and returns the next state. That feedback lets the planner revise its guidance when the robot gets stuck.

Visual: A visible planner message, Jev choice and probabilities, motor data.

## 092–122s · Tim · Watch a complete attempt

Here the robot is working through the coffee task. It has to approach the table, get within reach, collect the cup, and move toward the lectern. The task changes stage when the simulator registers the pickup. The cup follows the robot's hand while it travels. These are simulated interactions using an articulated robot and motion clips. The decision log remains visible, so we can connect the movement to the action that caused it.

Visual: Real coffee interaction and arrival; a command or task change.

## 122–153s · Vel · Inspect the recorded data

Opening an episode reveals the data behind the animation. Each step includes the robot's state, the selected action, a short label, the planner's guidance, and Jev's decision details. The record also includes whether the attempt succeeded, along with decision latency and model cost. Successful episodes are selected for training. Failed attempts stay in the records, where they help us inspect mistakes instead of disappearing from the comparison.

Visual: Open a genuine episode and its steps in Labeled data.

## 153–184s · Tim · Train on successful attempts

The training loop combines successful model-controlled attempts with a separately counted set of scripted demonstrations. The panel shows how many came from each source. A small neural network learns to map the numeric robot and goal state to an action. We split the recorded episodes into training and validation sets, save a checkpoint, and load its weights into the viewer. The cloud planners keep collecting attempts while later training rounds use the expanded dataset.

Visual: Actual training tab, source counts, loss curve and network.

## 184–216s · Vel · Run the local policy

The gold robot runs the learned policy. Its decisions are computed locally, without sending a request to a cloud model for each move. It can complete some attempts, and it still fails others. Those results are recorded through the same interface as the teachers. This policy receives numeric state and a goal; it does not understand speech or camera images. Watching its failures helps us decide what demonstrations and tests to add next.

Visual: Follow learned-bc, show zero tokens/cost and mixed results.

## 216–240s · Tim · What the prototype proves

The output of datanode is a set of inspectable robot demonstrations and a local control policy. This prototype uses kinematic simulation. Testing grasp forces and balance requires a physics model. Next, we would measure performance on new starting positions and layouts. The code and run records show how this experiment was produced.

Visual: Overview or stage, actual results and repository entrypoint.
