# ----------- IMPORTS -----------
import numpy as np
import matplotlib.pyplot as plt
import imageio as io
from scipy import ndimage
from skimage.color import rgb2gray
from scipy.ndimage import convolve
from skimage import filters

# ----------- LOADING AND DISPLAYING A TOY PROBLEM -----------
fileName = 'bad/frame-'
extension = '.png'
nx = 1092
ny = 292
N = 49 # nr. of frames
V = np.zeros((nx, ny, N))

for i in range(1, N+1):
    nr = str(i)
    V[:,:,i-1] = rgb2gray(io.imread(fileName + nr + extension))

def displayFrames(V, N, pauseTime):
    fig, ax = plt.subplots()
    for i in range(0, N):
        ax.cla()
        ax.imshow(V[:,:,i],cmap = plt.cm.gray)
        ax.set_title('Frame {}'.format(i+1))
        plt.pause(pauseTime)

pauseTime = 0.01
#displayFrames(V, N, pauseTime) # load the frames

# ----------- FRAME CHOICE -----------
chosenFrame = 35

# ----------- DISPLAY GRADIENT -----------
def displayGradient(Vx, Vy, Vt, gradientName, chosenFrame):
    fig, ax = plt.subplots(nrows = 1, ncols = 3)
    ax[0].imshow(Vx[:,:,chosenFrame],cmap = plt.cm.gray)
    ax[0].set_title('$V_x$')
    ax[1].imshow(Vy[:,:,chosenFrame],cmap = plt.cm.gray)
    ax[1].set_title('$V_y$')
    ax[2].imshow(Vt[:,:,chosenFrame],cmap = plt.cm.gray)
    ax[2].set_title('$V_t$')
    fig.tight_layout()
    plt.savefig(str(gradientName) + '-' + str(chosenFrame) + '.png', dpi = 300)
    plt.show()

# ----------- SOBEL GRADIENT CALCULATION -----------
Vx = ndimage.sobel(V, 1)
Vy = ndimage.sobel(V, 0)
Vt = ndimage.sobel(V, 2)

print(Vt.shape)

gradientName = 'SOBEL'
#displayGradient(Vx, Vy, Vt, gradientName, chosenFrame)

# ----------- SOLUTIONS TO THE LUCAS-KANADE SYSTEM -----------
p_x = 134 # x-position of the chosen pixel
p_y = 841 # y-position of the chosen pixel
p_t = chosenFrame # t-position of the chosen pixel

N = 21 # selected region (N x N)
A = np.empty((N*N, 2))
b = np.empty((N*N))

c = 0
for i in range(int(p_y-((N-1)/2)), int(p_y+((N-1)/2)+1)):
    for j in range(int(p_x-((N-1)/2)), int(p_x+((N-1)/2)+1)):
        A[c, :] = [Vx[i, j, chosenFrame], Vy[i, j, chosenFrame]]
        b[c] = -Vt[i, j, chosenFrame]
        c += 1
dr, residuals, rank, s = np.linalg.lstsq(A, b, rcond = None)

fig, ax = plt.subplots()
px, py = np.meshgrid(p_x, p_y)
ax.imshow(V[:,:,chosenFrame],cmap = plt.cm.gray)
ax.quiver(px, py, dr[0], dr[1], color = 'red')
plt.savefig('bad_' + str(N) + '.png', dpi = 300)
plt.show()

SET UP LUCAS-KANADE SYSTEM
N = 9 # selected region (N x N)
A = np.empty((N*N, 2))
b = np.empty((N*N))
dx = np.zeros((nx, ny, 38))
dy = np.zeros((nx, ny, 38))

c = 0
t = chosenFrame # a t loop can be easily constructed, it was avoided here to save time for verification purposes
for x in range(N, nx-N):
    for y in range(N, ny-N):
        for i in range(int(x-((N-1)/2)), int(x+((N-1)/2)+1)):
            for j in range(int(y-((N-1)/2)), int(y+((N-1)/2)+1)):
                A[c, :] = [Vx[i, j, t], Vy[i, j, t]]
                b[c] = -Vt[i, j, t]
                c += 1
        c = 0
        dr, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        dx[x, y, t] = dr[0]
        dy[x, y, t] = dr[1]

px = np.arange(N, nx-N, 50)
py = np.arange(N, ny-N, 50)
fig, ax = plt.subplots()
x, y = np.meshgrid(px, py)
ax.imshow(V[:,:,t],cmap = plt.cm.gray)
ax.quiver(x, y, dx[:, :, t][np.ix_(px,py)], dy[:, :, t][np.ix_(px,py)], color = 'red')
plt.savefig('friendly-' + str(t) + '.png')
plt.show()
