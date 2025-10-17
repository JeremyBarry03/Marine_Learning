const canvas = document.getElementById("nokey");
const ctx = canvas.getContext("2d");

const BUBBLE_COUNT = 30;
const bubbles = [];
let width = window.innerWidth;
let height = window.innerHeight;
let dpr = window.devicePixelRatio || 1;

function resizeCanvas() {
    width = window.innerWidth;
    height = window.innerHeight;
    dpr = window.devicePixelRatio || 1;

    canvas.width = width * dpr;
    canvas.height = height * dpr;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;

    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.scale(dpr, dpr);
}

function random(min, max) {
    return Math.random() * (max - min) + min;
}

function createBubble() {
    const radius = random(8, 36);
    const speed = random(0.35, 1.1);
    return {
        x: random(-radius, width + radius),
        y: height + radius + random(0, height),
        radius,
        speed,
        drift: random(0.08, 0.35),
        wobbleSpeed: random(1.0, 2.1),
        wobbleOffset: random(0, Math.PI * 2),
        baseHue: random(185, 210),
        life: 1,
        popping: false,
        popStart: 0,
        burstSeed: random(0.8, 1.2),
    };
}

function drawBubble(bubble, time) {
    const { x, y, radius, baseHue } = bubble;
    const fade = Math.max(0, bubble.life);

    // Bubble body gradient (gives a 3D look)
    const gradient = ctx.createRadialGradient(
        x - radius * 0.3,
        y - radius * 0.45,
        radius * 0.2,
        x,
        y,
        radius
    );
    gradient.addColorStop(0, `hsla(${baseHue}, 70%, 96%, ${0.45 * fade})`);
    gradient.addColorStop(0.5, `hsla(${baseHue + 5}, 65%, 68%, ${0.18 * fade})`);
    gradient.addColorStop(1, `hsla(${baseHue + 10}, 60%, 40%, ${0.03 * fade})`);

    ctx.fillStyle = gradient;
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fill();

    // Highlight
    const highlightX = x - radius * 0.45;
    const highlightY = y - radius * 0.55;
    const highlightGradient = ctx.createRadialGradient(
        highlightX,
        highlightY,
        0,
        highlightX,
        highlightY,
        radius * 0.55
    );
    highlightGradient.addColorStop(0, `rgba(255, 255, 255, ${0.28 * fade})`);
    highlightGradient.addColorStop(1, `rgba(255, 255, 255, 0)`);
    ctx.fillStyle = highlightGradient;
    ctx.beginPath();
    ctx.arc(
        highlightX,
        highlightY,
        radius * 0.6,
        0,
        Math.PI * 2
    );
    ctx.fill();

    // Rim sheen
    ctx.strokeStyle = `hsla(${baseHue}, 70%, 82%, ${0.12 * fade})`;
    ctx.lineWidth = Math.max(1, radius * 0.04);
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.stroke();

    // Soft inner sparkle
    const sparkleRadius = radius * 0.2;
    const sparkleGradient = ctx.createRadialGradient(
        x + radius * 0.25,
        y + radius * 0.1,
        0,
        x + radius * 0.25,
        y + radius * 0.1,
        sparkleRadius
    );
    sparkleGradient.addColorStop(0, `rgba(255, 255, 255, ${0.18 * fade})`);
    sparkleGradient.addColorStop(1, `rgba(255, 255, 255, 0)`);
    ctx.fillStyle = sparkleGradient;
    ctx.beginPath();
    ctx.arc(
        x + radius * 0.25,
        y + radius * 0.1,
        sparkleRadius,
        0,
        Math.PI * 2
    );
    ctx.fill();
}

function updateBubble(bubble, delta, time) {
    if (bubble.popping) {
        const elapsed = Math.max(0, time - bubble.popStart);
        bubble.life = Math.max(0, 1 - elapsed * 3);
        if (elapsed >= 0.45) {
            Object.assign(bubble, createBubble());
        }
        return;
    }

    const wobble = Math.sin(time * bubble.wobbleSpeed + bubble.wobbleOffset) * bubble.drift;
    bubble.x += wobble;
    bubble.y -= bubble.speed * delta * 0.06;

    if (bubble.y + bubble.radius < -20) {
        const reset = createBubble();
        bubble.x = reset.x;
        bubble.y = height + reset.radius;
        bubble.radius = reset.radius;
        bubble.speed = reset.speed;
        bubble.drift = reset.drift;
        bubble.wobbleSpeed = reset.wobbleSpeed;
        bubble.wobbleOffset = reset.wobbleOffset;
        bubble.baseHue = reset.baseHue;
    }
}

let lastTimestamp = 0;
function renderFrame(timestamp) {
    if (!lastTimestamp) {
        lastTimestamp = timestamp;
    }
    const delta = timestamp - lastTimestamp;
    lastTimestamp = timestamp;

    ctx.clearRect(0, 0, width, height);

    bubbles.forEach((bubble) => {
        updateBubble(bubble, delta, timestamp / 1000);
        drawBubble(bubble, timestamp / 1000);
        if (bubble.popping) {
            renderPopEffect(bubble, timestamp / 1000);
        }
    });

    window.requestAnimationFrame(renderFrame);
}

function renderPopEffect(bubble, time) {
    const elapsed = Math.max(0, time - bubble.popStart);
    const progress = Math.min(1, elapsed * 3 * bubble.burstSeed);

    const ringRadius = bubble.radius * (1 + progress * 1.9);
    const ringAlpha = Math.max(0, 0.3 * (1 - progress));
    const ringWidth = Math.max(0.8, bubble.radius * 0.15 * (1 - progress));

    ctx.strokeStyle = `hsla(${bubble.baseHue}, 85%, 85%, ${ringAlpha})`;
    ctx.lineWidth = ringWidth;
    ctx.beginPath();
    ctx.arc(bubble.x, bubble.y, ringRadius, 0, Math.PI * 2);
    ctx.stroke();

    const innerProgress = Math.min(1, elapsed * 4);
    const innerRadius = bubble.radius * (0.6 + innerProgress);
    const innerAlpha = Math.max(0, 0.18 * (1 - innerProgress));

    const innerGlow = ctx.createRadialGradient(
        bubble.x,
        bubble.y,
        innerRadius * 0.2,
        bubble.x,
        bubble.y,
        innerRadius
    );
    innerGlow.addColorStop(0, `hsla(${bubble.baseHue}, 90%, 95%, ${innerAlpha})`);
    innerGlow.addColorStop(1, `hsla(${bubble.baseHue}, 60%, 55%, 0)`);

    ctx.fillStyle = innerGlow;
    ctx.beginPath();
    ctx.arc(bubble.x, bubble.y, innerRadius, 0, Math.PI * 2);
    ctx.fill();
}

function popBubbleAt(x, y, time) {
    for (let i = bubbles.length - 1; i >= 0; i -= 1) {
        const bubble = bubbles[i];
        const dx = x - bubble.x;
        const dy = y - bubble.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        if (distance <= bubble.radius * 1.35) {
            bubble.popping = true;
            bubble.popStart = time;
            return;
        }
    }
}

canvas.addEventListener("click", (event) => {
    const rect = canvas.getBoundingClientRect();
    const x = (event.clientX - rect.left);
    const y = (event.clientY - rect.top);
    popBubbleAt(x, y, performance.now() / 1000);
});

function init() {
    resizeCanvas();
    bubbles.length = 0;
    for (let i = 0; i < BUBBLE_COUNT; i += 1) {
        bubbles.push(createBubble());
    }
    window.requestAnimationFrame(renderFrame);
}

window.addEventListener("resize", () => {
    resizeCanvas();
});

init();
