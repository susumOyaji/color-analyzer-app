document.addEventListener('DOMContentLoaded', () => {
    console.log('DOMContentLoaded fired');

    // Elements
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const analysisSection = document.getElementById('analysis-section');
    const imagePreview = document.getElementById('image-preview');
    const paletteGrid = document.getElementById('palette-grid');
    const colorCountSlider = document.getElementById('color-count-slider');
    const colorCountValue = document.getElementById('color-count-value');
    const toast = document.getElementById('toast');
    const changeImageBtn = document.getElementById('change-image-btn');
    const resetCropBtn = document.getElementById('reset-crop-btn');

    console.log('Elements:', { dropZone, fileInput, analysisSection, imagePreview });

    // State
    let currentImage = null;
    let cropper = null;
    const analyzeBtn = document.getElementById('analyze-btn');

    console.log('analyzeBtn:', analyzeBtn);

    if (!analyzeBtn) {
        console.error('analyzeBtn not found!');
    }

    // Event Listeners for Drag & Drop
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, unhighlight, false);
    });

    function highlight() {
        dropZone.classList.add('drag-over');
    }

    function unhighlight() {
        dropZone.classList.remove('drag-over');
    }

    dropZone.addEventListener('drop', handleDrop, false);
    dropZone.addEventListener('click', (e) => {
        console.log('dropZone clicked!');
        console.log('Click target:', e.target);
        console.log('fileInput:', fileInput);
        fileInput.click();
    }, true); // Use capture phase to catch event before children
    fileInput.addEventListener('change', handleFiles);

    console.log('Event listeners registered');

    // Slider Event
    colorCountSlider.addEventListener('input', (e) => {
        const count = e.target.value;
        colorCountValue.textContent = count;
    });

    // Analyze button event
    if (analyzeBtn) {
        analyzeBtn.addEventListener('click', () => {
            console.log('Analyze button clicked');
            if (currentImage) {
                performAnalysis();
            } else {
                console.error('No image loaded');
            }
        });
    } else {
        console.warn('Analyze button not found, event listener not registered');
    }

    // Scan button event
    const scanBtn = document.getElementById('scan-btn');
    if (scanBtn) {
        scanBtn.addEventListener('click', () => {
            console.log('Scan button clicked');
            if (currentImage) {
                performScan();
            } else {
                console.error('No image loaded');
            }
        });
    }

    // Change Image Button
    if (changeImageBtn) {
        changeImageBtn.addEventListener('click', () => {
            resetApp();
        });
    }

    // Reset Crop Button
    if (resetCropBtn) {
        resetCropBtn.addEventListener('click', () => {
            if (cropper) {
                cropper.reset();
            }
        });
    }

    function resetApp() {
        currentImage = null;
        if (cropper) {
            cropper.destroy();
            cropper = null;
        }
        imagePreview.src = '';
        fileInput.value = ''; // Reset file input

        analysisSection.classList.add('hidden');
        dropZone.classList.remove('hidden');
        dropZone.style.display = 'block'; // Ensure it's visible

        // Clear results
        paletteGrid.innerHTML = '';
        const colorBar = document.getElementById('color-bar');
        if (colorBar) colorBar.innerHTML = '';
        const scanChart = document.getElementById('scan-chart');
        if (scanChart) scanChart.innerHTML = '';
        const harmoniesGrid = document.getElementById('harmonies-grid');
        if (harmoniesGrid) harmoniesGrid.innerHTML = '';
    }


    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles({ target: { files: files } });
    }

    function handleFiles(e) {
        console.log('handleFiles called', e);
        const files = e.target.files;
        console.log('Files:', files);
        if (files.length > 0) {
            const file = files[0];
            console.log('File:', file);
            if (file.type.startsWith('image/')) {
                console.log('Valid image file detected');
                const reader = new FileReader();
                reader.onload = (e) => {
                    console.log('FileReader onload');
                    const img = new Image();
                    img.onload = () => {
                        console.log('Image onload');
                        currentImage = img;
                        imagePreview.src = img.src;

                        // Hide drop zone and show analysis section
                        dropZone.classList.add('hidden');
                        dropZone.style.display = 'none';
                        analysisSection.classList.remove('hidden');

                        // Initialize Cropper
                        if (cropper) {
                            cropper.destroy();
                        }
                        cropper = new Cropper(imagePreview, {
                            aspectRatio: NaN,
                            viewMode: 1,
                            autoCropArea: 1,
                            responsive: true,
                            background: false
                        });

                        // Scroll to analysis section smoothly
                        setTimeout(() => {
                            analysisSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
                        }, 100);
                    };
                    img.src = e.target.result;
                };
                reader.readAsDataURL(file);
            } else {
                showToast('Please upload a valid image file.');
            }
        }
    }

    // Perform analysis with crop data
    async function performAnalysis() {
        console.log('performAnalysis called');
        if (!cropper) {
            console.error('Cropper not initialized!');
            return;
        }

        const colorCount = colorCountSlider.value;

        // Get crop data
        const cropData = cropper.getData(true); // true = rounded values
        console.log('Crop Data:', cropData);

        // Create FormData to send image to server
        const formData = new FormData();

        let blob;
        try {
            const response = await fetch(imagePreview.src);
            blob = await response.blob();
        } catch (e) {
            console.error("Failed to fetch image blob from src", e);
            return;
        }

        formData.append('image', blob, 'image.jpg');
        formData.append('num_colors', colorCount);

        // Add crop coordinates
        formData.append('crop_x', Math.round(cropData.x));
        formData.append('crop_y', Math.round(cropData.y));
        formData.append('crop_w', Math.round(cropData.width));
        formData.append('crop_h', Math.round(cropData.height));

        console.log('Sending request to server...');

        try {
            const response = await fetch('http://localhost:5000/analyze', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error('Server analysis failed');
            }

            const data = await response.json();
            console.log('Server response:', data);
            const dominantColors = data.colors;

            // Calculate total pixels (sum of counts) for percentage calculation
            const totalPixels = dominantColors.reduce((sum, c) => sum + c.count, 0);

            renderPalette(dominantColors, totalPixels);
            renderColorBar(dominantColors, totalPixels);

            if (dominantColors.length > 0) {
                generateHarmonies(dominantColors[0]);
            }

        } catch (error) {
            console.error('Error analyzing image:', error);
            showToast('Error analyzing image. Is the server running?');
        }
    }

    // Perform scan analysis
    async function performScan() {
        console.log('performScan called');
        if (!cropper) {
            console.error('Cropper not initialized!');
            return;
        }

        const cropData = cropper.getData(true);
        console.log('Scan Crop Data:', cropData);

        const formData = new FormData();

        let blob;
        try {
            const response = await fetch(imagePreview.src);
            blob = await response.blob();
        } catch (e) {
            console.error("Failed to fetch image blob from src", e);
            return;
        }

        formData.append('image', blob, 'image.jpg');
        formData.append('num_slices', 10);
        formData.append('colors_per_slice', 3);

        // Add crop coordinates
        formData.append('crop_x', Math.round(cropData.x));
        formData.append('crop_y', Math.round(cropData.y));
        formData.append('crop_w', Math.round(cropData.width));
        formData.append('crop_h', Math.round(cropData.height));

        console.log('Sending scan request to server...');

        try {
            const response = await fetch('http://localhost:5000/scan', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error('Server scan failed');
            }

            const data = await response.json();
            console.log('Scan response:', data);

            renderScanChart(data.slices);

        } catch (error) {
            console.error('Error scanning image:', error);
            showToast('Error scanning image. Is the server running?');
        }
    }

    // Render scan chart
    function renderScanChart(slices) {
        const scanChartContainer = document.getElementById('scan-chart');
        if (!scanChartContainer) return;


        scanChartContainer.innerHTML = '';
        scanChartContainer.style.display = 'flex';
        scanChartContainer.style.gap = '0.5rem';
        scanChartContainer.style.marginBottom = '2rem';

        slices.forEach((slice, index) => {
            const column = document.createElement('div');
            column.style.flex = '1';
            column.style.display = 'flex';
            column.style.flexDirection = 'column';
            column.style.gap = '0.25rem';

            slice.colors.forEach(color => {
                const colorBlock = document.createElement('div');
                colorBlock.style.backgroundColor = `rgb(${color.r}, ${color.g}, ${color.b})`;
                colorBlock.style.height = '60px';
                colorBlock.style.borderRadius = '4px';
                colorBlock.style.cursor = 'pointer';
                colorBlock.title = color.hex;
                colorBlock.addEventListener('click', () => copyToClipboard(color.hex));
                column.appendChild(colorBlock);
            });

            scanChartContainer.appendChild(column);
        });
    }

    function renderPalette(colors, totalPixels) {
        paletteGrid.innerHTML = '';

        // Analyze button is now static in HTML


        colors.forEach((color, index) => {
            const percentage = ((color.count / totalPixels) * 100).toFixed(1);
            const hex = color.hex || rgbToHex(color.r, color.g, color.b);
            const rgbString = `rgb(${color.r}, ${color.g}, ${color.b})`;

            // Create Card
            const card = document.createElement('div');
            card.className = 'color-card';
            card.style.animationDelay = `${index * 0.1}s`;
            card.classList.add('fade-in-up');

            card.innerHTML = `
                <div class="color-preview" style="background-color: ${rgbString}">
                    <span class="percentage">${percentage}%</span>
                </div>
                <div class="color-info">
                    <div class="hex-code">${hex}</div>
                    <div class="rgb-code">${rgbString}</div>
                </div>
            `;

            card.addEventListener('click', () => {
                copyToClipboard(hex);
            });

            paletteGrid.appendChild(card);
        });
    }

    function renderColorBar(colors, totalPixels) {
        const colorBarContainer = document.getElementById('color-bar');

        if (!colorBarContainer) return;

        colorBarContainer.innerHTML = '';

        // Normalize to the sum of displayed colors so the bar is full width
        const totalDisplayedCount = colors.reduce((sum, c) => sum + c.count, 0);

        colors.forEach(color => {
            const percentage = (color.count / totalDisplayedCount) * 100;
            const hex = color.hex || rgbToHex(color.r, color.g, color.b);

            const segment = document.createElement('div');
            segment.className = 'color-bar-segment';
            segment.style.width = `${percentage}%`;
            segment.style.backgroundColor = `rgb(${color.r}, ${color.g}, ${color.b})`;
            segment.dataset.percentage = `${Math.round(percentage)}%`;
            segment.title = hex;

            segment.addEventListener('click', () => {
                copyToClipboard(hex);
            });

            colorBarContainer.appendChild(segment);
        });
    }

    function rgbToHex(r, g, b) {
        return "#" + ((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1).toUpperCase();
    }

    function copyToClipboard(text) {
        navigator.clipboard.writeText(text).then(() => {
            showToast(`Copied ${text} to clipboard!`);
        });
    }

    function showToast(message) {
        toast.textContent = message;
        toast.classList.add('show');
        setTimeout(() => {
            toast.classList.remove('show');
        }, 3000);
    }

    function generateHarmonies(baseColor) {
        const hsl = rgbToHsl(baseColor.r, baseColor.g, baseColor.b);
        const harmoniesContainer = document.getElementById('harmonies-grid');
        harmoniesContainer.innerHTML = '';

        const patterns = [
            { name: 'Complementary', colors: getComplementary(hsl) },
            { name: 'Analogous', colors: getAnalogous(hsl) },
            { name: 'Triadic', colors: getTriadic(hsl) },
            { name: 'Split Complementary', colors: getSplitComplementary(hsl) }
        ];

        patterns.forEach(pattern => {
            const patternEl = document.createElement('div');
            patternEl.className = 'harmony-group';

            const title = document.createElement('h3');
            title.className = 'harmony-title';
            title.textContent = pattern.name;
            patternEl.appendChild(title);

            const colorsContainer = document.createElement('div');
            colorsContainer.className = 'harmony-colors';

            pattern.colors.forEach(color => {
                const rgb = hslToRgb(color.h, color.s, color.l);
                const hex = rgbToHex(rgb.r, rgb.g, rgb.b);

                const colorBox = document.createElement('div');
                colorBox.className = 'harmony-color-box';
                colorBox.style.backgroundColor = hex;
                colorBox.title = hex;
                colorBox.addEventListener('click', () => copyToClipboard(hex));

                const hexLabel = document.createElement('span');
                hexLabel.className = 'harmony-hex';
                hexLabel.textContent = hex;
                colorBox.appendChild(hexLabel);

                colorsContainer.appendChild(colorBox);
            });

            patternEl.appendChild(colorsContainer);
            harmoniesContainer.appendChild(patternEl);
        });
    }

    function rgbToHsl(r, g, b) {
        r /= 255, g /= 255, b /= 255;
        const max = Math.max(r, g, b), min = Math.min(r, g, b);
        let h, s, l = (max + min) / 2;

        if (max === min) {
            h = s = 0; // achromatic
        } else {
            const d = max - min;
            s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
            switch (max) {
                case r: h = (g - b) / d + (g < b ? 6 : 0); break;
                case g: h = (b - r) / d + 2; break;
                case b: h = (r - g) / d + 4; break;
            }
            h /= 6;
        }
        return { h: h * 360, s: s * 100, l: l * 100 };
    }

    function hslToRgb(h, s, l) {
        h /= 360, s /= 100, l /= 100;
        let r, g, b;

        if (s === 0) {
            r = g = b = l; // achromatic
        } else {
            const hue2rgb = (p, q, t) => {
                if (t < 0) t += 1;
                if (t > 1) t -= 1;
                if (t < 1 / 6) return p + (q - p) * 6 * t;
                if (t < 1 / 2) return q;
                if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
                return p;
            };
            const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
            const p = 2 * l - q;
            r = hue2rgb(p, q, h + 1 / 3);
            g = hue2rgb(p, q, h);
            b = hue2rgb(p, q, h - 1 / 3);
        }
        return { r: Math.round(r * 255), g: Math.round(g * 255), b: Math.round(b * 255) };
    }

    function getComplementary(hsl) {
        return [
            hsl,
            { h: (hsl.h + 180) % 360, s: hsl.s, l: hsl.l }
        ];
    }

    function getAnalogous(hsl) {
        return [
            { h: (hsl.h - 30 + 360) % 360, s: hsl.s, l: hsl.l },
            hsl,
            { h: (hsl.h + 30) % 360, s: hsl.s, l: hsl.l }
        ];
    }

    function getTriadic(hsl) {
        return [
            hsl,
            { h: (hsl.h + 120) % 360, s: hsl.s, l: hsl.l },
            { h: (hsl.h + 240) % 360, s: hsl.s, l: hsl.l }
        ];
    }

    function getSplitComplementary(hsl) {
        return [
            hsl,
            { h: (hsl.h + 150) % 360, s: hsl.s, l: hsl.l },
            { h: (hsl.h + 210) % 360, s: hsl.s, l: hsl.l }
        ];
    }
});
