
function toggleMenu() {
    const sidebar = document.getElementById('sidebar');
    sidebar.classList.toggle('active');
}

let isPlaying = false;
let currentAudioFile = null;

function togglePlayPause() {
    const button = document.getElementById('play-pause');
    const audioPlayer = document.getElementById('audio-player');

    if (!currentAudioFile || !audioPlayer.src) {
        console.log("No audio file available");
        alert("No audio available. Please capture a gesture first.");
        return;
    }

    if (audioPlayer.paused) {
        audioPlayer.play()
            .then(() => {
                button.textContent = "Pause";
                isPlaying = true;
            })
            .catch(error => {
                console.error("Audio playback error:", error);
                alert("Error playing audio: " + error.message);
            });
    } else {
        audioPlayer.pause();
        button.textContent = "Play";
        isPlaying = false;
    }
}

document.getElementById('audio-player').addEventListener('ended', function() {
    document.getElementById('play-pause').textContent = "Play";
    isPlaying = false;
});

function clearText() {
    document.getElementById('text-output').value = "Text: Waiting for gestures...";
    const audioPlayer = document.getElementById('audio-player');
    audioPlayer.src = '';
    audioPlayer.pause();
    audioPlayer.style.display = 'none';
    document.getElementById('play-pause').textContent = "Play";
    currentAudioFile = null;
    isPlaying = false;
}

function updateCurrentPrediction() {
    fetch('/api/current_prediction')
        .then(response => response.json())
        .then(data => {
            const textOutput = document.getElementById('text-output');
            if (data.prediction && data.confidence > 0.3) {
                textOutput.value = `Text: ${data.prediction.toUpperCase()} (${(data.confidence * 100).toFixed(1)}%)`;
            } else {
                textOutput.value = "Text: Waiting for gestures...";
            }
        })
        .catch(error => {
            console.error('Error fetching prediction:', error);
        });
}

function captureWebcamPrediction() {
    const button = document.getElementById('capture-btn');
    const originalText = button.textContent;
    
    button.textContent = "Capturing...";
    button.disabled = true;

    fetch('/conversion', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: 'action=predict'
    })
    .then(response => {
        console.log('Response status:', response.status);
        if (!response.ok) throw new Error(`HTTP error! Status: ${response.status}`);
        return response.text();
    })
    .then(html => {
        console.log('Response HTML:', html);
        const parser = new DOMParser();
        const doc = parser.parseFromString(html, 'text/html');

        const errorElement = doc.querySelector('.error-message');
        if (errorElement && errorElement.textContent.trim()) {
            console.log('Error message:', errorElement.textContent);
            alert(errorElement.textContent);
            return;
        }

        const textElement = doc.querySelector('#text-result');
        const audioElement = doc.querySelector('#audio-result');

        if (textElement && textElement.textContent.trim()) {
            document.getElementById('text-output').value = `Text: ${textElement.textContent}`;
            console.log('Text result:', textElement.textContent);
        } else {
            console.log('No text result in response');
            document.getElementById('text-output').value = 'Text: No gesture detected';
        }

        if (audioElement && audioElement.textContent.trim()) {
            currentAudioFile = audioElement.textContent;
            const audioPlayer = document.getElementById('audio-player');
            audioPlayer.src = `/static/audio/${currentAudioFile}`;
            audioPlayer.style.display = 'block';
            console.log('Setting audio src:', audioPlayer.src);
            audioPlayer.play().then(() => {
                console.log('Audio playing:', currentAudioFile);
                document.getElementById('play-pause').textContent = 'Pause';
                isPlaying = true;
            }).catch(error => {
                console.error('Audio playback error:', error);
                alert('Error playing audio: ' + error.message);
            });
        } else {
            console.log('No audio result in response');
            alert('No audio generated. Please try again.');
        }
    })
    .catch(error => {
        console.error('Fetch error:', error);
        alert('Error capturing prediction: ' + error.message);
    })
    .finally(() => {
        button.textContent = originalText;
        button.disabled = false;
    });
}

document.addEventListener('DOMContentLoaded', function() {
    setInterval(updateCurrentPrediction, 500);
    const captureBtn = document.getElementById('capture-btn');
    if (captureBtn) {
        captureBtn.addEventListener('click', captureWebcamPrediction);
    }
    const hamburgerBtn = document.getElementById('hamburger');
    if (hamburgerBtn) {
        hamburgerBtn.addEventListener('click', toggleMenu);
    }
})
