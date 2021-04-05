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
    ax[0].imshow(Vx[:,:,chosenFrame],cmap = plt.cm.gray)
    ax[0].set_title('$V_x$')
    ax[1].imshow(Vy[:,:,chosenFrame],cmap = plt.cm.gray)
    ax[1].set_title('$V_y$')
    ax[2].imshow(Vt[:,:,chosenFrame],cmap = plt.cm.gray)
    ax[2].set_title('$V_t$')
    fig.tight_layout()
    plt.savefig(str(gradientName) + '-' + str(chosenFrame) + '.png', dpi = 300)
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

# ----------- FILTERS HEAD TO HEAD -----------
# fig, ax = plt.subplots(nrows = 1, ncols = 2)
# ax[0].imshow(sobel_x, cmap = plt.cm.gray)
# ax[0].set_title('Sobel')
# ax[1].imshow(prewitt_x, cmap = plt.cm.gray)
# ax[1].set_title('Prewitt')
# fig.tight_layout()
# plt.savefig('sobel.png', dpi = 300)
# plt.show()

# ----------- FLIPPING DIRECTION OF REFERENCE -----------
sobel_x = sobel_x*(-1) # -> right
sobel_y = sobel_y*(-1) # -> down

Vx = convolve(V[:, :, chosenFrame], sobel_x)
Vy = convolve(V[:, :, chosenFrame], sobel_y)

filterName = 'FLIPPED SOBEL 2D'
#display2Dfilter(Vx, Vy, filterName, chosenFrame)

# ----------- SOBEL GRADIENT CALCULATION -----------
Vx = ndimage.sobel(V, 1)
Vy = ndimage.sobel(V, 0)
Vt = ndimage.sobel(V, 2)

gradientName = 'SOBEL'
displayGradient(Vx, Vy, Vt, gradientName, chosenFrame)

# ----------- SOLUTIONS TO THE LUCAS-KANADE SYSTEM -----------
p_x = 48 # x-position of the chosen pixel
p_y = 128 # y-position of the chosen pixel
p_t = chosenFrame # t-position of the chosen pixel

N = 3 # selected region (N x N)

# SINGLE LOCAL REGION IN A SINGLE FRAME
A = np.empty((N*N, 2))
b = np.empty((N*N))

c = 0
for i in range(int(p_x-((N-1)/2)), int(p_x+((N-1)/2)+1)):
    for j in range(int(p_y-((N-1)/2)), int(p_y+((N-1)/2)+1)):
        A[c, :] = [Vx[i, j, chosenFrame], Vy[i, j, chosenFrame]]
        b[c] = -Vt[i, j, chosenFrame]
        c += 1
dr, residuals, rank, s = np.linalg.lstsq(A, b, rcond = None)

fig, ax = plt.subplots()
px, py = np.meshgrid(p_x, p_y)
ax.imshow(V[:,:,chosenFrame],cmap = plt.cm.gray)
ax.quiver(px, py, dr[0], dr[1], color = 'red')
plt.savefig('one_point_' + str(N) + '.png', dpi = 300)
plt.show()

# SET UP LUCAS-KANADE SYSTEM
A = np.empty((N*N, 2))
b = np.empty((N*N))
dx = np.zeros((nx, ny, 64))
dy = np.zeros((nx, ny, 64))

c = 0
t = 40 # a t loop can be easily constructed, it was avoided here to save time for verification purposes
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

px = np.arange(N, nx-N, 10)
py = np.arange(N, ny-N, 10)
fig, ax = plt.subplots()
x, y = np.meshgrid(px, py)
ax.imshow(V[:,:,t],cmap = plt.cm.gray)
ax.quiver(x, y, dx[:, :, t][np.ix_(px,py)], dy[:, :, t][np.ix_(px,py)], color = 'red')
plt.savefig('toy_frame_' + str(40) + '.png')
plt.show()
