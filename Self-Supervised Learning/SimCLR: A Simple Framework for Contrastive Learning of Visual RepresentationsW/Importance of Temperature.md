Remember your similarity function is the cosine distance, which with normalized vectors goes from -1 to 1. 
And remember that we are exping the result, with the resulting range going from 1/e to e.

Note how small that range is, and note that if the vectors were orthogonal to each other than the similarity is 0 and after exp(0)=1

So when the temperature=1, you are making the model try and make the negative pairs all be antiparallel to each other. 
Being (for example) orthogonal is not enough since the loss will still be quite high. 
That's obviously not ideal, if two vectors are orthogonal that should be enough of a separation.

Now set the temperature=0.1. 
Now the range goes from exp(-0.1)=4e-5 to exp(10)=22000 ! 
In case of orthogonality the similarity after exp is still 1 of course. 
But now making the vectors of a negative pair orthogonal (rather than close to parallel) will result in a much larger decrease in loss 
(if that's not clear remember we want to maximize a fraction where the negative pairs are in the bottom, positive pair in the top)!
