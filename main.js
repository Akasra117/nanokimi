// NanoLancer - Main JavaScript File
// Handles all interactive functionality, animations, and data management

class NanoLancer {
    constructor() {
        this.db = new NanoLancerDatabase();
        this.filteredTools = [];
        this.currentOffset = 0;
        this.limit = 20;
        this.isLoading = false;
        this.currentUser = null;
        
        this.init();
    }
    
    init() {
        this.initializeData();
        this.setupEventListeners();
        this.initializeAnimations();
        this.loadInitialContent();
        this.setupScrollAnimations();
    }
    
    // Initialize data from database
    initializeData() {
        this.tools = this.db.getAllTools();
        this.courses = this.db.getAllCourses();
        this.filteredTools = [...this.tools];
        
        // Initialize current user if in localStorage
        const userData = localStorage.getItem('currentUser');
        if (userData) {
            this.currentUser = JSON.parse(userData);
        }
    }
    
    // Setup event listeners
    setupEventListeners() {
        // Hero buttons
        document.getElementById('exploreToolsBtn')?.addEventListener('click', () => {
            document.getElementById('toolsSection').scrollIntoView({ behavior: 'smooth' });
        });
        
        document.getElementById('startLearningBtn')?.addEventListener('click', () => {
            window.location.href = 'courses.html';
        });
        
        // Search functionality
        document.getElementById('heroSearch')?.addEventListener('input', (e) => {
            this.handleSearch(e.target.value);
        });
        
        document.getElementById('toolsSearch')?.addEventListener('input', (e) => {
            this.handleSearch(e.target.value);
        });
        
        // Filter functionality
        document.querySelectorAll('.category-filter, .license-filter').forEach(filter => {
            filter.addEventListener('change', () => this.applyFilters());
        });
        
        document.getElementById('clearFilters')?.addEventListener('click', () => {
            this.clearAllFilters();
        });
        
        // Load more functionality
        document.getElementById('loadMoreBtn')?.addEventListener('click', () => {
            this.loadMoreTools();
        });
        
        // Authentication buttons
        document.getElementById('loginBtn')?.addEventListener('click', () => {
            this.showAuthModal('login');
        });
        
        document.getElementById('signupBtn')?.addEventListener('click', () => {
            this.showAuthModal('signup');
        });
    }
    
    // Initialize animations
    initializeAnimations() {
        // Animate hero text on load
        anime({
            targets: '.neon-text',
            opacity: [0, 1],
            translateY: [50, 0],
            duration: 1000,
            delay: 500,
            easing: 'easeOutExpo'
        });
        
        // Animate CTA buttons
        anime({
            targets: '#exploreToolsBtn, #startLearningBtn',
            opacity: [0, 1],
            translateY: [30, 0],
            duration: 800,
            delay: anime.stagger(200, {start: 1000}),
            easing: 'easeOutExpo'
        });
    }
    
    // Setup scroll animations
    setupScrollAnimations() {
        const observerOptions = {
            threshold: 0.1,
            rootMargin: '0px 0px -50px 0px'
        };
        
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('visible');
                }
            });
        }, observerOptions);
        
        document.querySelectorAll('.fade-in-up').forEach(el => {
            observer.observe(el);
        });
    }
    
    // Load initial content
    loadInitialContent() {
        this.renderTools();
        this.renderCourses();
        this.animateCounters();
    }
    
    // Render tools grid
    renderTools() {
        const toolsGrid = document.getElementById('toolsGrid');
        if (!toolsGrid) return;
        
        // If this is the first render (offset is 0), show initial batch
        // Otherwise, append new tools to existing ones
        if (this.currentOffset === 0) {
            const toolsToShow = this.filteredTools.slice(0, this.limit);
            toolsGrid.innerHTML = toolsToShow.map(tool => this.createToolCard(tool)).join('');
        } else {
            // Append new tools
            const newTools = this.filteredTools.slice(this.currentOffset, this.currentOffset + this.limit);
            const newToolsHTML = newTools.map(tool => this.createToolCard(tool)).join('');
            toolsGrid.insertAdjacentHTML('beforeend', newToolsHTML);
        }
        
        // Update load more button visibility
        const loadMoreBtn = document.getElementById('loadMoreBtn');
        if (loadMoreBtn) {
            if (this.currentOffset + this.limit >= this.filteredTools.length) {
                loadMoreBtn.style.display = 'none';
            } else {
                loadMoreBtn.style.display = 'block';
            }
        }
        
        // Add hover animations
        this.addCardAnimations();
    }
    
    // Create tool card HTML
    createToolCard(tool) {
        const licenseColor = {
            'free': 'bg-neon-green text-space-blue',
            'freemium': 'bg-electric-cyan text-space-blue',
            'opensource': 'bg-purple-500 text-white'
        };
        
        return `
            <div class="card-3d glass rounded-2xl p-6 hover:shadow-2xl transition-all duration-300 cursor-pointer" data-tool-id="${tool.id}">
                <div class="flex items-start justify-between mb-4">
                    <div class="w-12 h-12 rounded-lg overflow-hidden">
                        <img src="${tool.logo}" alt="${tool.name}" class="w-full h-full object-cover">
                    </div>
                    <span class="px-3 py-1 text-xs font-semibold rounded-full ${licenseColor[tool.license]}">
                        ${tool.license.charAt(0).toUpperCase() + tool.license.slice(1)}
                    </span>
                </div>
                
                <h3 class="font-orbitron font-bold text-xl mb-3 text-white hover:text-electric-cyan transition-colors">
                    ${tool.name}
                </h3>
                
                <p class="text-gray-300 text-sm mb-4 line-clamp-3">
                    ${tool.description}
                </p>
                
                <div class="flex flex-wrap gap-2 mb-4">
                    ${tool.tags.slice(0, 3).map(tag => `
                        <span class="px-2 py-1 bg-white bg-opacity-10 text-xs rounded-full text-gray-300">
                            ${tag}
                        </span>
                    `).join('')}
                </div>
                
                <div class="flex items-center justify-between">
                    <div class="flex items-center space-x-1">
                        <svg class="w-4 h-4 text-yellow-400" fill="currentColor" viewBox="0 0 20 20">
                            <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z"/>
                        </svg>
                        <span class="text-sm text-gray-300">${tool.popularity}</span>
                    </div>
                    
                    <button class="px-4 py-2 bg-gradient-to-r from-electric-cyan to-neon-green text-space-blue rounded-lg text-sm font-semibold hover:shadow-lg transition-all">
                        Try Now
                    </button>
                </div>
            </div>
        `;
    }
    
    // Add card animations
    addCardAnimations() {
        const cards = document.querySelectorAll('.card-3d');
        
        cards.forEach(card => {
            card.addEventListener('mouseenter', () => {
                anime({
                    targets: card,
                    scale: 1.05,
                    rotateY: 5,
                    rotateX: 5,
                    duration: 300,
                    easing: 'easeOutQuad'
                });
            });
            
            card.addEventListener('mouseleave', () => {
                anime({
                    targets: card,
                    scale: 1,
                    rotateY: 0,
                    rotateX: 0,
                    duration: 300,
                    easing: 'easeOutQuad'
                });
            });
        });
    }
    
    // Render courses carousel
    renderCourses() {
        const coursesList = document.getElementById('coursesList');
        if (!coursesList) return;
        
        coursesList.innerHTML = this.courses.map(course => this.createCourseCard(course)).join('');
        
        // Initialize Splide carousel
        if (typeof Splide !== 'undefined') {
            new Splide('#coursesCarousel', {
                type: 'loop',
                perPage: 3,
                perMove: 1,
                gap: '2rem',
                autoplay: true,
                interval: 4000,
                breakpoints: {
                    1024: { perPage: 2 },
                    640: { perPage: 1 }
                }
            }).mount();
        }
    }
    
    // Create course card HTML
    createCourseCard(course) {
        return `
            <li class="splide__slide">
                <div class="glass rounded-2xl overflow-hidden hover:shadow-2xl transition-all duration-300">
                    <div class="aspect-video overflow-hidden">
                        <img src="${course.thumbnail}" alt="${course.title}" class="w-full h-full object-cover hover:scale-105 transition-transform duration-300">
                    </div>
                    <div class="p-6">
                        <div class="flex items-center justify-between mb-3">
                            <span class="px-3 py-1 bg-neon-green text-space-blue text-xs font-semibold rounded-full">
                                ${course.difficulty}
                            </span>
                            <div class="flex items-center space-x-1">
                                <svg class="w-4 h-4 text-yellow-400" fill="currentColor" viewBox="0 0 20 20">
                                    <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z"/>
                                </svg>
                                <span class="text-sm text-gray-300">${course.rating}</span>
                            </div>
                        </div>
                        
                        <h3 class="font-orbitron font-bold text-xl mb-2 text-white hover:text-electric-cyan transition-colors">
                            ${course.title}
                        </h3>
                        
                        <p class="text-gray-400 text-sm mb-4">${course.provider}</p>
                        
                        <p class="text-gray-300 text-sm mb-4 line-clamp-2">
                            ${course.description}
                        </p>
                        
                        <div class="flex items-center justify-between">
                            <span class="text-sm text-gray-400">${course.duration}</span>
                            <button class="px-4 py-2 bg-gradient-to-r from-electric-cyan to-neon-green text-space-blue rounded-lg text-sm font-semibold hover:shadow-lg transition-all">
                                Enroll Free
                            </button>
                        </div>
                    </div>
                </div>
            </li>
        `;
    }
    
    // Handle search functionality
    handleSearch(query) {
        if (!query.trim()) {
            this.filteredTools = [...this.tools];
        } else {
            const searchTerm = query.toLowerCase();
            this.filteredTools = this.tools.filter(tool => 
                tool.name.toLowerCase().includes(searchTerm) ||
                tool.description.toLowerCase().includes(searchTerm) ||
                tool.tags.some(tag => tag.toLowerCase().includes(searchTerm))
            );
        }
        
        this.currentOffset = 0;
        this.renderTools();
    }
    
    // Apply filters
    applyFilters() {
        const categoryFilters = Array.from(document.querySelectorAll('.category-filter:checked'))
            .map(cb => cb.value);
        const licenseFilters = Array.from(document.querySelectorAll('.license-filter:checked'))
            .map(cb => cb.value);
        
        this.filteredTools = this.tools.filter(tool => {
            const categoryMatch = categoryFilters.length === 0 || categoryFilters.includes(tool.category);
            const licenseMatch = licenseFilters.length === 0 || licenseFilters.includes(tool.license);
            return categoryMatch && licenseMatch;
        });
        
        this.currentOffset = 0;
        this.renderTools();
    }
    
    // Clear all filters
    clearAllFilters() {
        document.querySelectorAll('.category-filter, .license-filter').forEach(cb => {
            cb.checked = true;
        });
        
        document.getElementById('toolsSearch').value = '';
        this.filteredTools = [...this.tools];
        this.currentOffset = 0;
        this.renderTools();
    }
    
    // Load more tools
    loadMoreTools() {
        if (this.isLoading) return;
        
        this.isLoading = true;
        document.getElementById('loadingIndicator').classList.remove('hidden');
        
        // Simulate API delay
        setTimeout(() => {
            this.currentOffset += this.limit;
            this.renderTools();
            
            this.isLoading = false;
            document.getElementById('loadingIndicator').classList.add('hidden');
        }, 1000);
    }
    
    // Animate counters
    animateCounters() {
        const counters = [
            { element: document.getElementById('toolsCount'), target: 500 },
            { element: document.getElementById('coursesCount'), target: 100 },
            { element: document.getElementById('usersCount'), target: 10000 }
        ];
        
        counters.forEach(counter => {
            if (counter.element) {
                anime({
                    targets: { count: 0 },
                    count: counter.target,
                    duration: 2000,
                    delay: 500,
                    easing: 'easeOutExpo',
                    update: function(anim) {
                        const value = Math.floor(anim.animatables[0].target.count);
                        counter.element.textContent = value >= 1000 ? 
                            (value / 1000).toFixed(0) + 'K+' : 
                            value + '+';
                    }
                });
            }
        });
    }
    
    // Show authentication modal
    showAuthModal(type) {
        // Create modal overlay
        const modal = document.createElement('div');
        modal.className = 'fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50';
        modal.innerHTML = `
            <div class="glass rounded-2xl p-8 max-w-md w-full mx-4">
                <div class="text-center mb-6">
                    <h2 class="font-orbitron font-bold text-2xl text-white mb-2">
                        ${type === 'login' ? 'Welcome Back' : 'Join NanoLancer'}
                    </h2>
                    <p class="text-gray-300">
                        ${type === 'login' ? 'Sign in to access your learning journey' : 'Start your AI learning adventure'}
                    </p>
                </div>
                
                <form class="space-y-4">
                    ${type === 'signup' ? `
                        <div>
                            <label class="block text-sm font-medium text-gray-300 mb-2">Full Name</label>
                            <input type="text" class="w-full px-4 py-3 bg-white bg-opacity-10 border border-white border-opacity-20 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-electric-cyan">
                        </div>
                    ` : ''}
                    <div>
                        <label class="block text-sm font-medium text-gray-300 mb-2">Email</label>
                        <input type="email" class="w-full px-4 py-3 bg-white bg-opacity-10 border border-white border-opacity-20 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-electric-cyan">
                    </div>
                    <div>
                        <label class="block text-sm font-medium text-gray-300 mb-2">Password</label>
                        <input type="password" class="w-full px-4 py-3 bg-white bg-opacity-10 border border-white border-opacity-20 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-electric-cyan">
                    </div>
                    
                    <button type="submit" class="w-full py-3 bg-gradient-to-r from-electric-cyan to-neon-green text-space-blue rounded-lg font-bold hover:shadow-lg transition-all">
                        ${type === 'login' ? 'Sign In' : 'Create Account'}
                    </button>
                </form>
                
                <div class="mt-6 text-center">
                    <button class="text-electric-cyan hover:text-white transition-colors" onclick="this.closest('.fixed').remove()">
                        Continue as Guest
                    </button>
                </div>
            </div>
        `;
        
        document.body.appendChild(modal);
        
        // Close modal on outside click
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                modal.remove();
            }
        });
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new NanoLancer();
});

// Add some utility functions for enhanced interactivity
window.addEventListener('load', () => {
    // Add smooth scrolling to all anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({ behavior: 'smooth' });
            }
        });
    });
    
    // Add loading animation for images
    document.querySelectorAll('img').forEach(img => {
        img.addEventListener('load', function() {
            this.style.opacity = '1';
        });
        img.style.opacity = '0';
        img.style.transition = 'opacity 0.3s ease';
    });
});