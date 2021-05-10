from experiment_builder import ExperimentBuilder
from meta_model import MetaLearner


def generate_snapshots(duration,num_frame,length,n,func):
    snapshots = []
    x=np.linspace(0,length,n)
    for t in np.linspace(0,duration,num_frame):
      snapshots.append(func(x,t))
    snapshots = np.array(snapshots)
    return snapshots

def F(x,t):
  return np.sin(x-t)


if __name__ == '__main__':
	model = MetaLearner()
	data=generate_snapshots(1,10000,1,512,F) #10000*512 snapshots

	train_loader = torch.utils.data.DataLoader(data)
	meta_system = ExperimentBuilder(model=model,data=data)

	meta_system.run_experiment()
