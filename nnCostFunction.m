function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

for i=1:m
	a1=[1; X(i:i,1:end)'];
	z2=Theta1*a1;
	a2=[1; sigmoid(z2)];
	z3=Theta2*a2;
	a3=sigmoid(z3);
	k=y(i,1);
	y_i=zeros(1,num_labels);
	y_i(1,k)=1.0;
	J=J+sum((1/m)*(-y_i*log(a3)-(1.-y_i)*log(1.-a3)));

	d3=a3-y_i';
	d2=Theta2'*d3;
	d2=d2(2:end,:);
	d2=d2.*sigmoidGradient(z2);
	Theta2_grad=Theta2_grad+d3*(a2)';
	Theta1_grad=Theta1_grad+d2*(a1)';
endfor

[Theta1_r,Theta1_c]=size(Theta1_grad);
[Theta2_r,Theta2_c]=size(Theta2_grad);
J=J+(lambda/(2*m))*(sum(sum(Theta1(:,2:end).^2))+sum(sum(Theta2(:,2:end).^2)));
Theta1_grad=(1/m).*Theta1_grad+[zeros(Theta1_r,1),((lambda/m).*Theta1)(:,2:end)];
Theta2_grad=(1/m).*Theta2_grad+[zeros(Theta2_r,1),((lambda/m).*Theta2)(:,2:end)];

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
