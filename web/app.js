// web/app.js

document.addEventListener('DOMContentLoaded', () => {
    // Initialize Lucide icons
    lucide.createIcons();

    const chatForm = document.getElementById('chat-form');
    const chatInput = document.getElementById('chat-input');
    const chatHistory = document.getElementById('chat-history');
    const modeToggle = document.getElementById('mode-toggle');
    const cameraSelect = document.getElementById('camera-select');
    const liveStreamImg = document.getElementById('live-stream-img');
    const fpsCounter = document.getElementById('fps-counter');

    // Modal elements
    const videoModal = document.getElementById('video-modal');
    const closeModalBtn = document.getElementById('close-modal-btn');
    const modalVideoPlayer = document.getElementById('modal-video-player');
    const modalVideoInfo = document.getElementById('modal-video-info');
    const modalSidebarClips = document.getElementById('modal-sidebar-clips');

    // Global store for current video clips
    let currentClips = [];

    // Modal Event Listeners
    closeModalBtn.addEventListener('click', closeVideoModal);

    // Close on ESC
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') closeVideoModal();
    });

    function closeVideoModal() {
        videoModal.classList.add('hidden');
        modalVideoPlayer.pause();
    }

    function openVideoModal(clips) {
        if (!clips || clips.length === 0) return;
        currentClips = clips;

        // Setup Primary
        playClipInModal(clips[0]);

        // Setup Sidebar (List all clips)
        modalSidebarClips.innerHTML = '';
        clips.forEach((clip, index) => {
            const btn = document.createElement('div');
            btn.className = 'secondary-clip';
            btn.innerHTML = `
                <div style="display:flex; flex-direction:column; gap:4px;">
                    <span style="font-weight:600; font-size:0.85rem;">Match ${index + 1}</span>
                    <span class="clip-time">⏱️ ${clip.time}</span>
                </div>
                <div style="display:flex; flex-direction:column; align-items:flex-end;">
                    <span class="clip-score">${(clip.score * 100).toFixed(0)}%</span>
                </div>
            `;
            btn.addEventListener('click', () => playClipInModal(clip));
            modalSidebarClips.appendChild(btn);
        });

        // Show Modal
        videoModal.classList.remove('hidden');
    }

    function playClipInModal(clip) {
        // Parse time mm:ss to seconds
        const tParts = clip.time.split(":");
        let startSec = 0;
        if (tParts.length === 2) {
            startSec = parseInt(tParts[0]) * 60 + parseInt(tParts[1]);
        }

        modalVideoPlayer.src = `/media/${clip.camera}#t=${startSec}`;
        modalVideoPlayer.play();
        modalVideoInfo.innerText = `Camera: ${clip.camera} | Match: ${(clip.score * 100).toFixed(0)}% | Time: ${clip.time}`;
    }

    // Handle Camera Change
    cameraSelect.addEventListener('change', (e) => {
        const camId = e.target.value;
        liveStreamImg.src = `/api/camera/stream/${camId}`;
        addSystemMessage(`Switched feed to ${camId}`);
    });

    // Handle Chat Submit
    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const query = chatInput.value.trim();
        if (!query) return;

        const mode = modeToggle.checked ? "both" : "text";

        // Add User message
        addUserMessage(query);
        chatInput.value = '';

        // Add loading bubble
        const loaderId = addLoader();

        try {
            const res = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query, mode })
            });

            const data = await res.json();
            removeLoader(loaderId);

            if (data.error) {
                addAiMessage(`Error: ${data.error}`, true);
            } else {
                renderAiResponse(data);
            }

        } catch (err) {
            removeLoader(loaderId);
            addAiMessage("Connection error. Could not reach VisionLog backend.", true);
        }
    });

    // ── UI Rendering Helpers ──────────────────────────────────────────────

    function addUserMessage(text) {
        const msg = document.createElement('div');
        msg.className = 'message message-user';
        msg.innerHTML = `
            <div class="avatar"><i data-lucide="user"></i></div>
            <div class="bubble"><p>${escapeHtml(text)}</p></div>
        `;
        chatHistory.appendChild(msg);
        lucide.createIcons({ root: msg });
        scrollToBottom();
    }

    function addSystemMessage(text) {
        const msg = document.createElement('div');
        msg.className = 'message message-ai';
        msg.innerHTML = `
            <div class="avatar" style="background: rgba(34, 197, 94, 0.2); color: var(--success);"><i data-lucide="info"></i></div>
            <div class="bubble" style="border-color: rgba(34, 197, 94, 0.3);"><p>${escapeHtml(text)}</p></div>
        `;
        chatHistory.appendChild(msg);
        lucide.createIcons({ root: msg });
        scrollToBottom();
    }

    function addAiMessage(text, isError = false) {
        const msg = document.createElement('div');
        msg.className = 'message message-ai';
        const colorStyle = isError ? 'style="border-color: var(--danger);"' : '';
        const icon = isError ? 'alert-triangle' : 'cpu';

        msg.innerHTML = `
            <div class="avatar"><i data-lucide="${icon}"></i></div>
            <div class="bubble" ${colorStyle}><p>${escapeHtml(text)}</p></div>
        `;
        chatHistory.appendChild(msg);
        lucide.createIcons({ root: msg });
        scrollToBottom();
    }

    function addLoader() {
        const id = 'loader-' + Date.now();
        const msg = document.createElement('div');
        msg.className = 'message message-ai';
        msg.id = id;
        msg.innerHTML = `
            <div class="avatar"><i data-lucide="cpu"></i></div>
            <div class="bubble">
                <div class="thinking-dots">
                    <div></div><div></div><div></div>
                </div>
            </div>
        `;
        chatHistory.appendChild(msg);
        lucide.createIcons({ root: msg });
        scrollToBottom();
        return id;
    }

    function removeLoader(id) {
        const el = document.getElementById(id);
        if (el) el.remove();
    }

    // ── Dual RAG Video Rendering ──────────────────────────────────────────

    function renderAiResponse(data) {
        const msg = document.createElement('div');
        msg.className = 'message message-ai';

        // 1. Convert Gemini markdown report to paragraphs
        let reportHtml = '';
        if (data.report) {
            const paragraphs = data.report.split('\n\n');
            reportHtml = paragraphs.map(p => `<p>${escapeHtml(p)}</p>`).join('');
        } else {
            // Fallback to raw text if it wasn't split properly
            reportHtml = `<p>${escapeHtml(data.raw_output || "Video evidence retrieved.")}</p>`;
        }

        // 2. Build Video Evidence DOM
        let videoHtml = '';
        if (data.video_evidence && data.video_evidence.length > 0) {
            // Save globally for the button click
            window._lastQueryClips = data.video_evidence;

            videoHtml += `
                <div class="evidence-container">
                    <button class="open-evidence-btn" onclick="openVideoModal(window._lastQueryClips)">
                        <i data-lucide="video"></i> View ${data.video_evidence.length} Video Matches
                    </button>
                </div>
            `;
        }

        // Combine
        msg.innerHTML = `
            <div class="avatar"><i data-lucide="cpu"></i></div>
            <div class="bubble">
                ${reportHtml}
                ${videoHtml}
            </div>
        `;

        chatHistory.appendChild(msg);

        // Re-init icons inside the new message
        lucide.createIcons({ root: msg });
        scrollToBottom();

        // 3. Auto-pop the modal overlay as the user requested
        if (data.video_evidence && data.video_evidence.length > 0) {
            // Slight delay to ensure DOM is ready
            setTimeout(() => {
                openVideoModal(data.video_evidence);
            }, 200);
        }
    }

    window.openVideoModal = openVideoModal; // Expose globally for the onclick handler

    function scrollToBottom() {
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }

    function escapeHtml(unsafe) {
        return unsafe
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    // Fake FPS updater just for visuals
    setInterval(() => {
        const jitter = (Math.random() * 2 - 1).toFixed(1);
        fpsCounter.innerText = (29.5 + parseFloat(jitter)).toFixed(1) + " FPS";
    }, 1000);
});
