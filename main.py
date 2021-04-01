# ----------- IMPORTS -----------
import numpy as np
import matplotlib.pyplot as plt
import imageio as io
from scipy import ndimage
from skimage.color import rgb2gray
from scipy.ndimage import convolve
from skimage import filters


# ----------- LOADING AND DISPLAYING A TOY PROBLEM -----------
fileName = 'toyProblem/frame_'
extension = '.png'
nx = 256
ny = 256
N = 64 # nr. of frames
V = np.zeros((nx, ny, N))

for i in range(1, N+1):
    if i <= 9:
        nr = "0" + str(i)
    else:
        nr = str(i)

    V[:,:,i-1] = rgb2gray(io.imread(fileName + nr + extension)) # load the frames

def displayFrames(V, N, pauseTime):
    fig, ax = plt.subplots()
    for i in range(0, N):
        ax.cla()
        ax.imshow(V[:,:,i],cmap = plt.cm.gray)
        ax.set_title('Frame {}'.format(i+1))
        plt.pause(pauseTime)

pauseTime = 0.01
#displayFrames(V, N, pauseTime)

# ----------- FRAME CHOICE -----------
chosenFrame = 6

# ----------- LOW LEVEL GRADIENT -----------
def lowLevelGradient(V):
    Vx = V[1:, :, :] - V[0:-1, :, :]
    Vy = V[:, 1:, :] - V[:, 0:-1, :]
    Vt = V[:, :, 1:] - V[:, :, 0:-1]
    return Vx, Vy, Vt

# image volume (nx, ny, N) -> gradient volume (nx - 1, ny - 1, N - 1)

Vx, Vy, Vt = lowLevelGradient(V)

# ----------- DISPLAY GRADIENT -----------
def displayGradient(Vx, Vy, Vt, gradientName, chosenFrame):
    fig, ax = plt.subplots(nrows = 1, ncols = 3)
    fig.suptitle('Frame ' + str(chosenFrame + 1) + ' (' + gradientName +')')
    ax[0].imshow(Vx[:,:,chosenFrame],cmap = plt.cm.gray)
    ax[0].set_title('$V_x$')
    ax[1].imshow(Vy[:,:,chosenFrame],cmap = plt.cm.gray)
    ax[1].set_title('$V_y$')
    ax[2].imshow(Vt[:,:,chosenFrame],cmap = plt.cm.gray)
    ax[2].set_title('$V_t$')
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    plt.show()

gradientName = 'LOW LEVEL GRADIENT'
#displayGradient(Vx, Vy, Vt, gradientName, chosenFrame)

# ----------- PREWITT KERNELS (2D) -----------
prewitt_x = np.array([[1, 0, -1],[1, 0, -1],[1, 0, -1]]) # -> left
prewitt_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]) # -> up
# or simply: prewitt_y = np.transpose(prewitt_x)

Vx = convolve(V[:, :, chosenFrame], prewitt_x)
Vy = convolve(V[:, :, chosenFrame], prewitt_y)

def display2Dfilter(Vx, Vy, filterName, chosenFrame):
    fig, ax = plt.subplots(nrows = 1, ncols = 2)
    fig.suptitle('Frame ' + str(chosenFrame + 1) + ' (' + filterName + ')')
    ax[0].imshow(Vx, cmap = plt.cm.gray)
    ax[0].set_title('$V_x$')
    ax[1].imshow(Vy, cmap = plt.cm.gray)
    ax[1].set_title('$V_y$')
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    plt.show()

filterName = 'PREWITT 2D'
#display2Dfilter(Vx, Vy, filterName, chosenFrame)

# ----------- SOBEL KERNELS (2D) -----------
sobel_x = np.array([[1, 0, -1],[2, 0, -2],[1, 0, -1]]) # -> right
sobel_y = np.array([[1, 2, 1],[0, 0, 0],[-1, -2, -1]]) # -> down

Vx = convolve(V[:, :, chosenFrame], sobel_x)
Vy = convolve(V[:, :, chosenFrame], sobel_y)

filterName = 'SOBEL 2D'
#display2Dfilter(Vx, Vy, filterName, chosenFrame)

# ----------- FLIPPING DIRECTION OF REFERENCE -----------
sobel_x = sobel_x*(-1) # -> right
sobel_y = sobel_y*(-1) # -> down

Vx = convolve(V[:, :, chosenFrame], sobel_x)
Vy = convolve(V[:, :, chosenFrame], sobel_y)

filterName = 'FLIPPED SOBEL 2D'
#display2Dfilter(Vx, Vy, filterName, chosenFrame)

# ----------- SOBEL GRADIENT CALCULATION -----------
Vx = filters.sobel(V, axis = 0)
Vy = filters.sobel(V, axis = 1)
Vt = filters.sobel(V, axis = 2)

gradientName = 'SOBEL'
#displayGradient(Vx, Vy, Vt, gradientName, chosenFrame)

# ----------- SOLUTIONS TO THE LUCAS-KANADE SYSTEM -----------
p_x = 180 # x-position of the chosen pixel
p_y = 130 # y-position of the chosen pixel
p_t = chosenFrame # t-position of the chosen pixel

N = 3 # selected region (N x N)

# SET UP LUCAS-KANADE SYSTEM
A = np.empty((N*N, 2))
b = np.empty((N*N))
dx = np.zeros((nx, ny, 64))
dy = np.zeros((nx, ny, 64))

c = 0
for t in range(0, 64):
    for x in range(1, nx-1):
        for y in range(1, ny-1):
            for i in range(int(x-((N-1)/2)), int(x+((N-1)/2)+1)):
                for j in range(int(y-((N-1)/2)), int(y+((N-1)/2)+1)):
                    A[c, :] = [Vx[i, j, t], Vy[i, j, t]]
                    b[c] = -Vt[i, j, t]
                    c += 1
            c = 0
            dx[x, y, t] = np.linalg.lstsq(A, b, rcond=None)[0][0]
            dy[x, y, t] = np.linalg.lstsq(A, b, rcond=None)[0][1]

fig, ax = plt.subplots()
x, y = np.meshgrid(np.arange(1, nx - 1, 10), np.arange(1, ny - 1, 10))
for i in range(0, 64):
    ax.cla()
    ax.imshow(V[:,:,i],cmap = plt.cm.gray)
    ax.quiver(x, y, dx[1:-1:10, 1:-1:10, i], dy[1:-1:10, 1:-1:10, i])
    ax.set_title('frame {}'.format(i+1))
    plt.pause(1)
