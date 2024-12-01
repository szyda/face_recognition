const video = document.getElementById('videoElement');

navigator.mediaDevices.getUserMedia({ video: true })
    .then((stream) => {
        video.srcObject = stream;
    })
    .catch((error) => {
        console.error("Error accessing the camera: ", error);
    });

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
    .then(response => response.json())
    .then(data => {
        displayResult(data);
    })
    .catch((error) => {
        console.error('Error:', error);
    });
}

function displayResult(data) {
    const resultMessage = document.getElementById('resultMessage');
    if (data.status === 'authorized') {
        resultMessage.textContent = 'Authorized';
        resultMessage.className = 'authorized';
    } else {
        resultMessage.textContent = 'Unauthorized';
        resultMessage.className = 'unauthorized';
    }
    resultMessage.style.display = 'block';
}
