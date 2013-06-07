function h5write(data, filename, varargin)

if nargin > 2
   if varargin{1}
      data=permute(data,[fliplr(1:(ndims(data)-1)) ndims(data)]);
   end
else
    data = permute(data,fliplr(1:ndims(data)));
end

if isreal(data)
    hdf5write(filename, '/real',data);
else
    hdf5write(filename, 'imag', imag(data),'WriteMode','overwrite');
    hdf5write(filename, 'real', real(data),'WriteMode','append');
    
end