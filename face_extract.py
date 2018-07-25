import numpy as np
import cv2
import os
##########################################
def distance(v1, v2):
    # Eucledian distance
    return np.sqrt(((v1-v2)**2).sum())

def knn(train, test, k=3):
    dist = []
    
    for i in range(train.shape[0]):
        # Get the vector and label
        ix = train[i, :-1]
        iy = train[i, -1]
        # Compute the distance from test point
        d = distance(test, ix)
        dist.append([d, iy])
    # Sort based on distance and get top k
    dk = sorted(dist, key=lambda x: x[0])[:k]
    # Retrieve only the labels
    labels = np.array(dk)[:, -1]
    
    # Get frequencies of each label
    output = np.unique(labels, return_counts=True)
    # Find max frequency and corresponding label
    index = np.argmax(output[1])
    return output[0][index]
##########################################    
# Initialize camera
cap = cv2.VideoCapture(0)

# Load the haar cascade for frontal face
face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_alt.xml')


dataset_path = '../facedata'

#data preperation

face_data =[]
labels =[]
names =[]
class_id =0

for fx in os.listdir(dataset_path) :
	if endswitch('.npy'):
		names.append(fx ,data_item.shape[0])

    face_data.append(data_item)

    target = class_id *np.ones((data_item.shape[0],-1))
    class_id +=1 
    label.append(target)

np.one	


while True:
	ret, frame = cap.read()
	if ret == False:
		continue
	# Convert frame to grayscale
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Detect multi faces in the image
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	k = 1

	# faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)

	# update the frame number
	

	for face in faces :
		x, y, w, h = face

		# Get the face ROI
		offset = 7
		face_section = frame[y-offset:y+h+offset, x-offset:x+w+offset]
		face_section = cv2.resize(face_section, (100, 100))
		
		# if skip % 10 == 0:
		# 	face_data.append(face_section)
		# 	print len(face_data)

		# Display the face ROI
		cv2.imshow(str(k), face_section)
		k += 1

		# Draw rectangle in the original image
		cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

	cv2.imshow("Faces", frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# # Convert face list to numpy array
# face_data = np.asarray(face_data)
# face_data = face_data.reshape((face_data.shape[0], -1))
# print face_data.shape

# # Save the dataset in filesystem
# np.save(dataset_path + file_name, face_data)
# print "Dataset saved at: {}".format(dataset_path + file_name + '.npy')

cv2.destroyAllWindows()