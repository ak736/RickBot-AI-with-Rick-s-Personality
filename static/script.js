// static/script.js
document.addEventListener('DOMContentLoaded', function() {
    // Add initial greeting
    addMessage("Wubba Lubba Dub Dub! I'm the smartest bot in the multiverse! *burp*", false);
});

function addMessage(message, isUser) {
    const chatMessages = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
    messageDiv.textContent = message;
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function clearChat() {
    const chatMessages = document.getElementById('chatMessages');
    chatMessages.innerHTML = '';
    // Add back the initial greeting
    addMessage("Wubba Lubba Dub Dub! I'm the smartest bot in the multiverse! *burp*", false);
}

async function sendMessage() {
    const userInput = document.getElementById('userMessage');
    const message = userInput.value.trim();
    
    if (message) {
        // Disable input while processing
        userInput.disabled = true;
        
        // Add user message
        addMessage(message, true);
        userInput.value = '';
        
        try {
            const response = await fetch('/get_response', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: message })
            });
            
            const data = await response.json();
            
            // Handle special commands
            if (data.command === 'quit') {
                window.location.href = '/';
                return;
            } else if (data.command === 'clear') {
                clearChat();
            } else {
                addMessage(data.response, false);
            }
            
        } catch (error) {
            console.error('Error:', error);
            addMessage('*burp* Error in dimension C-137!', false);
        } finally {
            // Re-enable input
            userInput.disabled = false;
            userInput.focus();
        }
    }
}

// Allow Enter key to send message
document.getElementById('userMessage').addEventListener('keypress', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});