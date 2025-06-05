window.scrollToBottom = (elementId) => {
    var element = document.getElementById(elementId);
    if (element) {
        element.scrollTop = element.scrollHeight;
    } else {
        console.log("Element not found: " + elementId);
    }
};
