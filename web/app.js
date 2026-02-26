document.addEventListener('DOMContentLoaded', () => {
    const camSelect = document.getElementById('cam-select');
    const videoStream = document.getElementById('video-stream');
    const placeholder = document.getElementById('video-placeholder');
    const feedStatusText = document.getElementById('feed-status-text');
    const chatContext = document.getElementById('chat-context');
    
    const chatHistory = document.getElementById('chat-history');
    const chatInput = document.getElementById('chat-input');
    const sendBtn = document.getElementById('send-btn');

    // --- 1. Load Cameras ---
    async function loadCameras() {
        try {
            const res = await fetch('/api/cameras');
            const data = await res.json();
            
            data.cameras.forEach(cam => {
                const opt = document.createElement('option');
                opt.value = cam.id;
                opt.textContent = `${cam.id}: ${cam.name}`;
                camSelect.appendChild(opt);
            });

            // Trigger initial state
            handleCameraChange();
        } catch (e) {
            console.error("Failed to load cameras:", e);
            feedStatusText.textContent = "ERROR: BACKEND OFFLINE";
            feedStatusText.style.color = "var(--danger)";
        }
    }

    // --- 2. Feed Management ---
    function handleCameraChange() {
        const selectedCam = camSelect.value;
        
        if (selectedCam === "ALL") {
            videoStream.src = "";
            videoStream.classList.add("hidden");
            placeholder.classList.remove("hidden");
            placeholder.textContent = "FACILITY OVERVIEW MAP (COMING SOON)";
            feedStatusText.textContent = "STANDBY: MULTI-VIEW";
            feedStatusText.style.color = "var(--warning)";
            chatContext.textContent = "CTX: FACILITY [ALL]";
        } else {
            // Load MJPEG Stream
            videoStream.src = `/api/video/${selectedCam}`;
            videoStream.classList.remove("hidden");
            placeholder.classList.add("hidden");
            feedStatusText.textContent = `LIVE: ${selectedCam}`;
            feedStatusText.style.color = "var(--success)";
            chatContext.textContent = `CTX: ${selectedCam}`;
        }
    }

    camSelect.addEventListener('change', handleCameraChange);

    // --- 3. Chat Logic ---
    function addMessage(text, isUser = false, isError = false) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${isUser ? 'user-msg' : 'system-msg'}`;
        
        const icon = document.createElement('div');
        icon.className = 'msg-icon';
        icon.textContent = isUser ? 'USR' : 'AI';

        const bubble = document.createElement('div');
        bubble.className = 'msg-bubble';
        if (isError) bubble.style.color = 'var(--danger)';
        
        // Handle basic markdown bold (**text**)
        const formattedText = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        
        // Split on newlines to create <p> tags
        formattedText.split('\n').forEach(line => {
            if (line.trim()) {
                const p = document.createElement('p');
                p.innerHTML = line; // Use innerHTML to allow the <strong> tags
                bubble.appendChild(p);
            }
        });

        msgDiv.appendChild(icon);
        msgDiv.appendChild(bubble);
        chatHistory.appendChild(msgDiv);
        
        // Auto scroll
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }

    async function sendQuery() {
        const query = chatInput.value.trim();
        if (!query) return;

        const activeCam = camSelect.value;
        
        // Add user message
        addMessage(query, true);
        chatInput.value = "";
        
        // Loading indicator
        const loadingId = "loader-" + Date.now();
        const loadingDiv = document.createElement('div');
        loadingDiv.className = 'message system-msg';
        loadingDiv.id = loadingId;
        loadingDiv.innerHTML = `
            <div class="msg-icon">AI</div>
            <div class="msg-bubble"><p class="loading-dots">Analyzing surveillance records</p></div>
        `;
        chatHistory.appendChild(loadingDiv);
        chatHistory.scrollTop = chatHistory.scrollHeight;

        try {
            const res = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: query, camera_id: activeCam })
            });
            const data = await res.json();
            
            // Remove loader
            document.getElementById(loadingId).remove();
            
            // Add AI response
            addMessage(data.reply, false, data.error);
        } catch (e) {
            document.getElementById(loadingId).remove();
            addMessage("Communication failure. RAG backend unreachable.", false, true);
        }
    }

    sendBtn.addEventListener('click', sendQuery);
    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendQuery();
        }
    });

    // --- Init ---
    loadCameras();
});
