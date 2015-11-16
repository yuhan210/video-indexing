import sys
sys.path.append('/home/t-yuche/mcdnn/caffe/python')
import caffe
from caffe.proto import caffe_pb2
import google.protobuf.text_format


def generate_db():

      

if __name__ == "__main__":

    generate_db()
    
    solver_param = caffe_pb2.SolverParameter()
    with open('./models/solver_template.prototxt') as f:
        google.protobuf.text_format.Merge(f.read(), solver_param)

    # nitem 
    solver_param.stepsize = nitem
    solver_param.max_iter = int(nitem * 2.5) 
    
    with open('./models/solver_template_temp.prototxt', 'w') as f:
        f.write(google.protobuf.text_format.MessageToString(solver_param))
    name = f.name

    solver = caffe.SGDSolver(name)
    solver.solve()

    trained_model = str(solver_param.snapshot_prefix) + '_iter_' + str(solver_param.max_iter) + '.caffemodel'

     
