function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);
p = zeros(size(X, 1), 1);

for i=1:m
	x=[1; X(i:i,1:end)'];
	a2=[1; sigmoid(Theta1*x)];
	a3=sigmoid(Theta2*a2);
	[maximum,maxIndex]=max(a3);
	p(i,1)=maxIndex;
endfor

end
