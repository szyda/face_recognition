const video = document.getElementById('videoElement');
const verificationPage = document.getElementById('verificationPage');
const resultPage = document.getElementById('resultPage');
const resultText = document.getElementById('resultText');
const backButton = document.getElementById('backButton');

function startCamera() {
    navigator.mediaDevices.getUserMedia({ video: true })
        .then((stream) => {
            video.srcObject = stream;
        })
        .catch((error) => {
            console.error("Error accessing the camera: ", error);
            alert("Error accessing the camera. Please allow camera access and try again.");
        });
}

function stopCamera() {
    if (video.srcObject) {
        video.srcObject.getTracks().forEach(track => track.stop());
        video.srcObject = null;
    }
}

startCamera();

document.getElementById('verifyButton').addEventListener('click', () => {
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    const context = canvas.getContext('2d');

    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    const imageDataURL = canvas.toDataURL('image/jpeg');
    verifyIdentity(imageDataURL);
});

function verifyIdentity(imageDataURL) {
    fetch('http://localhost:5000/verify', {
        method: 'POST',
        body: JSON.stringify({ image: imageDataURL }),
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(errorData => {
                throw new Error(errorData.message || 'Server error');
            });
        }
        return response.json();
    })
    .then(data => {
        displayResult(data);
    })
    .catch((error) => {
        console.error('Error:', error);
        displayResult({ status: 'error', message: error.message });
    });
}

function displayResult(data) {
    stopCamera();
    verificationPage.style.display = 'none';
    resultPage.style.display = 'block';

    if (data.status === 'authorized') {
        resultText.textContent = 'Authorized';
        resultText.style.color = 'green';
    } else if (data.status === 'unauthorized') {
        resultText.textContent = 'Unauthorized';
        resultText.style.color = 'red';
    } else if (data.status === 'error') {
        resultText.textContent = `Error: ${data.message}`;
        resultText.style.color = 'red';
    }
}

backButton.addEventListener('click', () => {
    startCamera();
    verificationPage.style.display = 'block';
    resultPage.style.display = 'none';
    resultText.textContent = '';
});
