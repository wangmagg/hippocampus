import sys
import pickle

def toByu():
    with open(sys.argv[1], "rb") as input:
        V = pickle.load(input)

    with open(sys.argv[2], "rb") as input:
        F = pickle.load(input)

    file = open(sys.argv[3], "w")

    num_points = V.shape[0]
    num_faces = F.shape[0]

    file.write("1 %d %d %d \n" % (num_points, num_faces, num_faces*3))
    file.write("1 %d \n" % num_faces)

    for v in V:
        file.write("%f %f %f \n" % (v[0], v[1], v[2]))

    for i,f in enumerate(F):
        if i == num_faces - 1:
            file.write("%d %d %d" % (f[0]+1, f[1]+1, -1 * (f[2]+1)))
        else:
            file.write("%d %d %d \n" % (f[0]+1, f[1]+1, -1*(f[2]+1)))

    file.close()

