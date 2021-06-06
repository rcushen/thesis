using DelimitedFiles,Statistics,LinearAlgebra,Plots

## Declare functions

#ansatz functions, x₀ is the centre point and x is the query point
function ϕ(ϵ,x₀,x) 
    #x₀ must be a column vector if calling with x₀ a single vector
    stencil=[0 0; 0 2π ; 0 -2π ; 2π 0 ; -2π 0]
    s=zeros(size(x,1))
    for l=1:size(x,1)
        for c=1:5,
        s[l]=s[l]+exp(-norm(x₀-(x[l,:]+stencil[c,:]))^2/ϵ^2)
        end
    end
    return s
end

#linear combinations of ansatz functions, α are coefficients, X₀ is an nx2 array of data/centre points
function f(α,ϵ,X₀,x)
    v=zeros(size(x,1))
    for i=1:length(α)
        v=v+α[i]*ϕ(ϵ,X₀[i,:],x)
    end
    return v
end

#collocation matrix, XX is an nx2 array of data/centre points, xx is an nx2 array of query points
function Mf(XX,xx,ϵ)
    M=zeros(size(XX,1),size(xx,1))
    for i=1:size(XX,1)
            M[i,:]=ϕ(ϵ,XX[i,:],xx)
    end
    return M
end

#distance matrix, pairwise distances between data points in an nx2 array xx
function Df(xx)
    D=zeros(size(xx,1),size(xx,1))
    stencil=[0 0; 0 2π ; 0 -2π ; 2π 0 ; -2π 0]
    for i=1:size(xx,1)
        for j=1:size(xx,1)
            s=zeros(5)
            for c=1:5,
                s[c]=norm(xx[i,:]-(xx[j,:]+stencil[c,:]))
            end
            D[i,j]=minimum(s)
        end
    end
    return D
end

X=readdlm("x.dat")

D=Df(X)

## build distance statistics on the data points

kthneighdist(k)=sort(D,dims=2)[:,k]
meanNNdist=mean(kthneighdist(2))
maxNNdist=maximum(kthneighdist(2))
sortedmeandist=mean(sort(D,dims=2))

#decide on ϵ from above statistics
ϵ=0.16 

## create "test centres" for the least squares

xunif=0:2π/50:2π*(49/50)
yunif=xunif
Xunif=[repeat(xunif,1,length(yunif))'[:] repeat(yunif,length(xunif),1)[:]]

## compute weights
M=Mf(Xunif,X[1:2000,:],ϵ)
#w2000_12, r2000_12, v2000_12=nnlsq([M2000_12a ; ones(1,2000)],[ones(2401,1)*π*(0.2)^2; (2π)^2],0)
w, r, v=nnlsq(M,ones(size(Xunif)[1],1)*π*ϵ^2,0)

## plotting scripts

xfine=0:2π/100:2π
yfine=xfine

plotly()

#contour plot
xyfine=[repeat(xfine,1,length(yfine))'[:] repeat(yfine,length(xfine),1)[:]]
zfine=f(w,ϵ,X,xyfine)
contourf(xyfine[:,1],xyfine[:,2],zfine,levels=50,c=:jet,clim=(0,0.4))

#surface plot
Xfine=repeat(xfine,1,length(yfine))
Yfine=repeat(yfine,1,length(xfine))'

Zfine=zeros(size(Xfine))

for k=1:length(xfine)
    for l=1:length(yfine)
        Zfine[k,l]=f(w,ϵ,X,[Xfine[k,l] Yfine[k,l]])[1]
    end
end
surface(Xfine,Yfine,Zfine)



