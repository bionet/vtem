function A = h5read(filename, varargin)

if nargin < 1
    [filename,pathname]=uigetfile({'*.h5'},'Select The Input File');
    filename=[pathname filename];
end

info = hdf5info(filename);

if length(info.GroupHierarchy.Datasets) == 2
    I = hdf5read(info.GroupHierarchy.Datasets(1));
    R = hdf5read(info.GroupHierarchy.Datasets(2));
    A = complex(R,I);
else
    A = hdf5read(info.GroupHierarchy.Datasets);
end

if(nargin>1)
   if varargin{1}
     A=permute(A,[fliplr(1:(ndims(A)-1)) ndims(A)]);
   end
else
   A=permute(A,fliplr([1:ndims(A)]));
end

 