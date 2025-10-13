var canvas = document.getElementById('nokey'),
    can_w = parseInt(canvas.getAttribute('width')),
    can_h = parseInt(canvas.getAttribute('height')),
    ctx = canvas.getContext('2d');

var BALL_NUM = 30;
var R = 0.8;
var balls = [];
var link_line_width = 0.8;
var dis_limit = 260;
var alpha_f = 0.03;

// Random color generator
function getRandomColor() {
    var r = Math.floor(Math.random() * 256);
    var g = Math.floor(Math.random() * 256);
    var b = Math.floor(Math.random() * 256);
    return 'rgb(' + r + ',' + g + ',' + b + ')';
}

// Random speed
function getRandomSpeed(pos) {
    var min = -1,
        max = 1;
    switch (pos) {
        case 'top':
            return [randomNumFrom(min, max), randomNumFrom(0.1, max)];
        case 'right':
            return [randomNumFrom(min, -0.1), randomNumFrom(min, max)];
        case 'bottom':
            return [randomNumFrom(min, max), randomNumFrom(min, -0.1)];
        case 'left':
            return [randomNumFrom(0.1, max), randomNumFrom(min, max)];
        default:
            return [0, 0];
    }
}

function randomNumFrom(min, max) {
    return Math.random() * (max - min) + min;
}

function randomSidePos(length) {
    return Math.ceil(Math.random() * length);
}

// Random Ball
function getRandomBall() {
    var pos = ['top', 'right', 'bottom', 'left'][Math.floor(Math.random() * 4)];
    return {
        x: pos === 'right' ? can_w + R : pos === 'left' ? -R : randomSidePos(can_w),
        y: pos === 'top' ? -R : pos === 'bottom' ? can_h + R : randomSidePos(can_h),
        vx: getRandomSpeed(pos)[0],
        vy: getRandomSpeed(pos)[1],
        r: R,
        alpha: 1,
        phase: randomNumFrom(0, 10),
        color: getRandomColor()
    };
}

// Draw Ball
function renderBalls() {
    balls.forEach(function (b) {
        ctx.fillStyle = b.color;
        ctx.beginPath();
        ctx.arc(b.x, b.y, R, 0, Math.PI * 2, true);
        ctx.closePath();
        ctx.fill();
    });
}

// Update balls
function updateBalls() {
    balls.forEach(function (b) {
        b.x += b.vx;
        b.y += b.vy;

        // Wrap around the screen
        b.x = b.x > can_w ? -R : b.x < -R ? can_w : b.x;
        b.y = b.y > can_h ? -R : b.y < -R ? can_h : b.y;

        // Update color
        if (Math.random() < 0.01) {
            b.color = getRandomColor();
        }

        // Update alpha
        b.phase += alpha_f;
        b.alpha = Math.abs(Math.cos(b.phase));
    });
}

// Draw lines
function lerpColor(color1, color2, fraction) {
    var c1 = color1.substring(4, color1.length - 1).split(',');
    var c2 = color2.substring(4, color2.length - 1).split(',');
    var r = Math.round(lerp(Number(c1[0]), Number(c2[0]), fraction));
    var g = Math.round(lerp(Number(c1[1]), Number(c2[1]), fraction));
    var b = Math.round(lerp(Number(c1[2]), Number(c2[2]), fraction));
    return 'rgb(' + r + ',' + g + ',' + b + ')';
}

function lerp(start, end, fraction) {
    return start + (end - start) * fraction;
}

function renderLines() {
    for (var i = 0; i < balls.length; i++) {
        for (var j = i + 1; j < balls.length; j++) {
            var fraction = getDisOf(balls[i], balls[j]) / dis_limit;
            if (fraction < 1) {
                var alpha = (1 - fraction).toString();
                var color = lerpColor(balls[i].color, balls[j].color, fraction);
                ctx.strokeStyle = color;
                ctx.lineWidth = link_line_width;
                ctx.globalAlpha = alpha;
                ctx.beginPath();
                ctx.moveTo(balls[i].x, balls[i].y);
                ctx.lineTo(balls[j].x, balls[j].y);
                ctx.stroke();
                ctx.closePath();
            }
        }
    }
    ctx.globalAlpha = 1;
}

// Distance calculator
function getDisOf(b1, b2) {
    var delta_x = Math.abs(b1.x - b2.x);
    var delta_y = Math.abs(b1.y - b2.y);
    return Math.sqrt(delta_x * delta_x + delta_y * delta_y);
}

// Add new balls if needed
function addBallIfy() {
    if (balls.length < BALL_NUM) {
        balls.push(getRandomBall());
    }
}

// Main render function
function render() {
    ctx.clearRect(0, 0, can_w, can_h);
    renderBalls();
    renderLines();
    updateBalls();
    addBallIfy();
    window.requestAnimationFrame(render);
}

// Initialize canvas and balls
function initCanvas() {
    canvas.setAttribute('width', window.innerWidth);
    canvas.setAttribute('height', window.innerHeight);
    can_w = parseInt(canvas.getAttribute('width'));
    can_h = parseInt(canvas.getAttribute('height'));
}

function initBalls(num) {
    for (var i = 1; i <= num; i++) {
        balls.push(getRandomBall());
    }
}

// Start the animation
function goMovie() {
    initCanvas();
    initBalls(BALL_NUM);
    window.requestAnimationFrame(render);
}

// Handle window resize
window.addEventListener('resize', function () {
    initCanvas();
});

// Start the animation
goMovie();
