import sys
import numpy as np

def toMesh():
    # Open byu file
    with open(sys.argv[1], "r") as fp:
        line = fp.readline()
        els = line.strip().split()

        num_points = int(els[1])
        num_faces = int(els[2])

        V = np.zeros((num_points, 3))
        F = np.zeros((num_faces, 3))

        fp.readline()

        for i in range(num_points):
            line = fp.readline()
            els = line.strip().split()
            V[i] = [float(els[0]), float(els[1]), float(els[2])]

        for i in range(num_faces):
            line = fp.readline()
            els = line.strip().split()
            F[i] = [int(els[0])-1, int(els[1])-1, -1*(int(els[2]))-1]

    with open(sys.argv[2], "wb") as output:
        pickle.dump(torch.as_tensor(V, dtype = torch.float32), output)

    with open(sys.argv[3], "rb") as output:
        pickle.dump(torch.as_tensor(F, dtype = torch.long), output)

if __name__ == "__main__":
    toMesh()

