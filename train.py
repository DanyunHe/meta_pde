from experiment_builder import ExperimentBuilder
from meta_model import MetaLearner

model = MetaLearner()
## TODO
data = # get snapshot (m*n)

train_loader = torch.utils.data.DataLoader(data)
meta_system = ExperimentBuilder(model=model,data=data)

meta_system.run_experiment()
