require 'nn'

dofile 'models/unsupervised_model_simple.lua'

num_enc = 3

model = nn.Sequential()
pooling, enc = Encoder(3,64)
model:add(enc)
for i=1,num_enc-1 do
	pooling, enc = Encoder(64*math.pow(2,i-1), 64*math.pow(2,i))
	model:add(enc)
end

-- Put a linear view on top of the last Encoder
linear_s = (64*96*96) / (math.pow(2,num_enc+1))
model:add(nn.View(linear_s))

-- Add a Classifier
classifier = nn.Sequential()
classifier:add(nn.Dropout(0.5))
classifier:add(nn.Linear(linear_s,512))
classifier:add(nn.BatchNormalization(512))
classifier:add(nn.ReLU(true))
classifier:add(nn.Dropout(0.5))
classifier:add(nn.Linear(512,10))
model:add(classifier)
return model

