function [dataset] = narma(systemorder, nts, nr_samples)
 len = nts;
 
 dataset = struct('inputs', cell(1,nr_samples), 'outputs', cell(1,nr_samples));
 s = systemorder - 1;
 
 for i = 1:nr_samples
     %Create random input sequence
     dataset(i).inputs = rand(1,len)/2;
     dataset(i).outputs = 0.1*ones(1,len);
     for n=systemorder+1:len-1
         dataset(i).outputs(n+1) = .2*dataset(i).outputs(n) + ...
             .004*dataset(i).outputs(n)*sum(dataset(i).outputs(n-s:n)) + ...
             1.5*dataset(i).inputs(n-s) * dataset(i).inputs(n) + 0.001;
     end
 end