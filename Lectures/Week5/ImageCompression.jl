using Images # Image loading, saving, manipulation
using HTTP   # Internet access
using Plots  # Precompiles on every startup (~20 secondss)
gr() # Needs modules Plots and GR to be installed, may need a rebuild of GR with ']build GR'
default(size=(400, 300), fmt = :png) # Default plot size, change output format to png

## <https://en.wikipedia.org/wiki/Image_compression Image compression> by k-means clustering
# <https://en.wikipedia.org/wiki/Channel_(digital_image)#RGB_Images
# 3-channel (RGB) color images> in 24-bit color representation allows for
# up to more than 16 million ($2^{24}$) different colors.
#
# In the computer, an RGB image is stored as a stack of (3) matrices:
#
# <<https://cdn.tutsplus.com/active/uploads/legacy/tuts/076_rgbShift/Tutorial/1.jpg>>
#
# By using the k-means clustering, we want to find a few (k) useful color
# combinations for representing & viewing an image.
#
# See also <http://stanford.edu/class/ee103/lectures/k-means_slides.pdf
# these notes (on clustering applications)>.

## Load and display image before pixel-clustering

# Load image stored on a local folder adress:
# myimage = "C:/Users/ulfin/Dropbox/MATH310/Forelesninger_2021/Clustering/Everyones_a_little_bit_racist_sometimes.jpg";
# Ximg = load(myimage);

# Load image from the Internet:
# ---------------------------------
#imageadress = "https://cdn.images.express.co.uk/img/dynamic/151/590x/secondary/spacex-launch-why-starman-tesla-roadster-david-bowie-falcon-heavy-1225205.jpg";
#imageadress = "http://pressarchive.theoldglobe.org/_img/pressphotos/pre2008%20photos/aveQ5.jpg";
#imageadress = "https://vgc.no/drfront/images/2018/02/12/c=1114,366,1920,1048;w=262;h=143;384858.jpg";
imageadress = "https://www.dagbladet.no/images/73342156.jpg?imageId=73342156&x=15.602322206096&y=10.807860262009&cropw=72.060957910015&croph=61.764705882353&width=912&height=521&compression=80";
# ---------------------------------
#imageadress = "https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png";
#imageadress = "http://www.johnloomis.org/ece563/notes/basics/components/mandrill/Mandrill.jpg";
# ---------------------------------
                                 # download("http://pressarchive.theoldglobe.org/_img/pressphotos/pre2008%20photos/aveQ5.jpg", "myimage.png"); Ximg = load("myimage.png");
myimage = download(imageadress); # Needs package ImageMagick
Ximg = load(myimage);

original = plot(Ximg, title = "The original image to be compressed by k-means color clustering") #, size = (1000,420))
display(original)
## Find image size and reshape to prepare pixel-data for clustering
n,m = size(Ximg); nm = n*m;


mat = channelview(Ximg); # Convert from image format to 3 x n x m (0-1).

# By using the <https://se.mathworks.com/help/matlab/ref/reshape.html
# reshape function in MATLAB> we can make a three-column matrix X so that
# each line in X is the RGB-vector of a pixel position of the image (Ximg).
X = float( reshape( permutedims(mat, (2,3,1)), (nm, 3) ) ); # Channels last, vectorize image dims, convert to float

# +
include("mykmeans.jl")
## Cluster RGB-pixel values (in X) into k color clusters by the k-means algorithm
k = 16; # The number of clusters

@time begin
Cid, Ccenters, J, cs = mykmeans(X,k); # This will take some time...
                                    # Cid:      a vector of cluster labels for the rows in X.
                                    # Ccenters: the resulting k cluster centers.
                                    # J:        the clustering objective function values
                                    # cs:       the cluster sizes (number of members in each cluster)
                                    ## Ccent = uint8(Ccenters); # Convert cluster centers into uint8-format
end # Time spent on the clustering process

# Plotting the objective function values reflecting the clustering process
Jplot = plot(J, linestyle = :dashdot, title = "Objective function (J) values",
    ylabel = "J (mean squared distace)", xlabel = "Clustering process interations", label = "J", size = (500, 300))
display(Jplot)
# Plotting the cluster sizes:
csplot = plot(cs, line = (:dot, 1), marker = ([:hex :d], 3, 0.8, Plots.stroke(3, :gray)), title = "Cluster sizes", label = "", size = (500, 300))
display(csplot)
## Reshape and display cluster labels into associated image
cl = reshape(Cid,(n,m)); # cl is an image (n x m - matrix) viewing the cluster labels
labelplot = plot(Gray.(cl/k), title = string("Image view of the pixelwise cluster labels for ", k, " clusters"), size = (1000,420))
display(labelplot)

# Compressed image using only the colors associated with the cluster centers:
print(string("The compressed image based on k=", k, " color clusters"))
# Use cluster-IDs (Cid) as lookup in cluster centers (Ccenters), reshape, permute and convert to RGB
cmpplot = plot(colorview(RGB, permutedims( reshape(Ccenters[Cid,:],(n, m, 3)), (3,1,2))), title = string("The compressed image based on k=", k, " color clusters"), size = (1000,420))
display(cmpplot)

## The residual image (difference between original - clustered result)
print(string("The residual image based on k=", k, " color clusters"))
resplot = plot(Ximg-colorview(RGB, permutedims( reshape(Ccenters[Cid,:],(n, m, 3)), (3,1,2))), title = string("The residual image based on k=", k, " color clusters"), size = (1000,420))
display(resplot)
