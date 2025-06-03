function toggleMenu() {
    document.getElementById('sidebar').classList.toggle('active');
}

let isPlaying = false;
function togglePlayPause() {
    const button = document.getElementById('play-pause');
    if (!isPlaying) {
        console.log("Play button clicked - No audio connected");
        button.textContent = "Pause";
        isPlaying = true;
    } else {
        console.log("Pause button clicked - No audio connected");
        button.textContent = "Play";
        isPlaying = false;
    }
}

function clearText() {
    document.getElementById('text-output').value = "Text: Waiting for gestures...";
    const audio = document.getElementById('audio-player');
    audio.src = '';
    isPlaying = false;
    document.getElementById('play-pause').textContent = "Play";
}