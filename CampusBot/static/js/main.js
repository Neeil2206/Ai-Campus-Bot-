/**
 * CampusBot Landing Page JavaScript
 * Handles navbar status indicator, mobile drawer, and smooth navigation.
 */

document.addEventListener("DOMContentLoaded", () => {
    initNavbarStatus();
    initMobileDrawer();
    initSmoothScroll();
});

async function initNavbarStatus() {
    const dot = document.getElementById("status-dot");
    const label = document.getElementById("status-label");
    const badge = document.getElementById("nav-status-badge");

    try {
        const res = await fetch("/api/status");
        if (!res.ok) throw new Error();
        const data = await res.json();

        if (data.ollama_online && data.chat_model_available) {
            if (dot) dot.className = "status-pulse-dot status-online";
            if (label) label.textContent = "Local AI Online";
            if (badge) badge.title = `Ollama ${data.chat_model} is connected and ready.`;
        } else if (data.ollama_online && !data.chat_model_available) {
            if (dot) dot.className = "status-pulse-dot status-warning";
            if (label) label.textContent = "Model Missing";
            if (badge) badge.title = `Ollama is running, but ${data.chat_model} is missing.`;
        } else {
            if (dot) dot.className = "status-pulse-dot status-offline";
            if (label) label.textContent = "Local AI Offline";
            if (badge) badge.title = "CampusBot's local AI service is unavailable. Please start Ollama.";
        }
    } catch (e) {
        if (dot) dot.className = "status-pulse-dot status-offline";
        if (label) label.textContent = "AI Offline";
    }
}

function initMobileDrawer() {
    const toggleBtn = document.getElementById("mobile-toggle-btn");
    const closeBtn = document.getElementById("drawer-close-btn");
    const drawer = document.getElementById("mobile-drawer");
    const links = document.querySelectorAll(".drawer-link");

    if (toggleBtn && drawer) {
        toggleBtn.addEventListener("click", () => {
            drawer.classList.add("open");
        });
    }

    if (closeBtn && drawer) {
        closeBtn.addEventListener("click", () => {
            drawer.classList.remove("open");
        });
    }

    links.forEach((link) => {
        link.addEventListener("click", () => {
            if (drawer) drawer.classList.remove("open");
        });
    });
}

function initSmoothScroll() {
    const navLinks = document.querySelectorAll(".nav-link[href^='#'], .drawer-link[href^='#']");
    navLinks.forEach((link) => {
        link.addEventListener("click", (e) => {
            const targetId = link.getAttribute("href");
            if (targetId && targetId !== "#") {
                const targetEl = document.querySelector(targetId);
                if (targetEl) {
                    e.preventDefault();
                    targetEl.scrollIntoView({ behavior: "smooth" });
                }
            }
        });
    });
}
