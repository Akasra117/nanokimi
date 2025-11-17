// NanoLancer Database - Enhanced AI Tools and Courses
// Comprehensive database with detailed descriptions and student management

class NanoLancerDatabase {
    constructor() {
        this.initializeData();
    }
    
    initializeData() {
        // Expanded AI Tools Database with 5-line descriptions
        this.tools = [
            {
                id: 1,
                name: "ChatGPT",
                category: "nlp",
                license: "freemium",
                region: "Global",
                tags: ["chatbot", "nlp", "openai", "conversation", "text-generation"],
                url: "https://chat.openai.com",
                logo: "resources/ai-tools/chatgpt-icon.png",
                popularity: 95,
                featured: true,
                description: `Advanced conversational AI model for natural language processing tasks, content generation, and intelligent conversations.
Built on GPT-4 architecture with enhanced reasoning capabilities and multimodal input support.
Supports code generation, creative writing, problem-solving, and educational explanations across multiple domains.
Features include conversation memory, custom instructions, and plugin ecosystem for extended functionality.
Widely used for customer service, content creation, programming assistance, and educational purposes.`,
                rating: 4.8,
                users: "100M+",
                pricing: "Free tier available, ChatGPT Plus at $20/month"
            },
            {
                id: 2,
                name: "Midjourney",
                category: "vision",
                license: "freemium",
                region: "Global",
                tags: ["image-generation", "art", "creative", "vision", "diffusion"],
                url: "https://midjourney.com",
                logo: "resources/ai-tools/midjourney-icon.png",
                popularity: 92,
                featured: true,
                description: `AI-powered image generation tool that creates stunning visuals from text prompts with artistic flair and high quality.
Utilizes advanced diffusion models to generate photorealistic images, digital art, and creative illustrations.
Offers fine control over style, composition, lighting, and artistic elements through detailed prompt engineering.
Features include image upscaling, variations, inpainting, and style transfer capabilities.
Popular among artists, designers, marketers, and content creators for visual content production.`,
                rating: 4.7,
                users: "15M+",
                pricing: "Basic plan starts at $10/month"
            },
            {
                id: 3,
                name: "TensorFlow",
                category: "ml",
                license: "opensource",
                region: "Global",
                tags: ["machine-learning", "google", "python", "neural-networks", "deep-learning"],
                url: "https://tensorflow.org",
                logo: "resources/ai-tools/tensorflow-icon.png",
                popularity: 88,
                featured: true,
                description: `Open-source machine learning platform for building and deploying ML models at scale with comprehensive ecosystem.
Developed by Google Brain team, supports deep learning, neural networks, and traditional ML algorithms.
Provides extensive APIs for Python, JavaScript, and mobile platforms with GPU acceleration support.
Features include TensorBoard for visualization, TensorFlow Serving for deployment, and TensorFlow Lite for mobile.
Widely adopted in research, industry, and education for AI development and production deployment.`,
                rating: 4.6,
                users: "10M+",
                pricing: "Completely free and open source"
            },
            {
                id: 4,
                name: "GitHub Copilot",
                category: "dev",
                license: "freemium",
                region: "Global",
                tags: ["code-generation", "github", "developer", "autocomplete", "programming"],
                url: "https://github.com/features/copilot",
                logo: "resources/ai-tools/copilot-icon.png",
                popularity: 90,
                featured: false,
                description: `AI-powered code completion tool that suggests code snippets and entire functions in real-time within your IDE.
Built on OpenAI Codex model, trained on billions of lines of public code from GitHub repositories.
Supports multiple programming languages including Python, JavaScript, TypeScript, Go, Ruby, and more.
Features include context-aware suggestions, function generation, test case creation, and documentation assistance.
Integrates with VS Code, JetBrains IDEs, Vim, and other popular development environments.`,
                rating: 4.5,
                users: "5M+",
                pricing: "Individual plan at $10/month, free for students"
            },
            {
                id: 5,
                name: "Claude AI",
                category: "nlp",
                license: "freemium",
                region: "Global",
                tags: ["ai-assistant", "safety", "reasoning", "anthropic", "conversation"],
                url: "https://claude.ai",
                logo: "resources/ai-tools/chatgpt-icon.png",
                popularity: 86,
                featured: false,
                description: `Anthropic's AI assistant focused on safety and helpful conversations with advanced reasoning capabilities.
Built with Constitutional AI methodology emphasizing helpfulness, harmlessness, and honesty in responses.
Features long conversation memory, complex reasoning, and nuanced understanding of context and ethics.
Supports multiple file uploads, code analysis, mathematical problem-solving, and creative writing tasks.
Designed with strong safety measures and ethical guidelines for responsible AI interaction.`,
                rating: 4.6,
                users: "2M+",
                pricing: "Free tier available, Claude Pro at $20/month"
            },
            {
                id: 6,
                name: "Stable Diffusion",
                category: "vision",
                license: "opensource",
                region: "Global",
                tags: ["image-generation", "open-source", "local", "stable-diffusion", "art"],
                url: "https://stability.ai",
                logo: "resources/ai-tools/midjourney-icon.png",
                popularity: 84,
                featured: false,
                description: `Open-source text-to-image generation model that creates high-quality images locally on consumer hardware.
Released by Stability AI, enables artists and developers to run AI image generation without cloud dependencies.
Supports various fine-tuned models, custom training, and extensive control over generation parameters.
Features include inpainting, outpainting, image-to-image translation, and style transfer capabilities.
Popular for custom AI art generation, character design, and creative content production.`,
                rating: 4.4,
                users: "3M+",
                pricing: "Free and open source, cloud services available"
            },
            {
                id: 7,
                name: "Hugging Face",
                category: "nlp",
                license: "opensource",
                region: "Global",
                tags: ["models", "nlp", "transformers", "community", "ml-platform"],
                url: "https://huggingface.co",
                logo: "resources/ai-tools/tensorflow-icon.png",
                popularity: 87,
                featured: false,
                description: `Platform for hosting and sharing machine learning models with easy-to-use APIs and extensive model hub.
Offers thousands of pre-trained models for NLP, computer vision, audio processing, and multimodal tasks.
Provides Transformers library with support for PyTorch, TensorFlow, and JAX frameworks.
Features include model hosting, inference APIs, model training, and collaborative model development.
Essential resource for AI researchers, developers, and organizations building ML applications.`,
                rating: 4.7,
                users: "1M+",
                pricing: "Free tier available, paid plans for enterprise"
            },
            {
                id: 8,
                name: "DALL-E 3",
                category: "vision",
                license: "freemium",
                region: "Global",
                tags: ["image-generation", "openai", "creative", "art", "text-to-image"],
                url: "https://openai.com/dall-e-3",
                logo: "resources/ai-tools/midjourney-icon.png",
                popularity: 89,
                featured: true,
                description: `Advanced AI image generation model that creates highly detailed and realistic images from text descriptions.
Latest iteration of OpenAI's DALL-E series with improved understanding of prompts and higher image quality.
Features enhanced text rendering, better aspect ratio handling, and more accurate interpretation of complex descriptions.
Supports various artistic styles, photorealistic imagery, and creative visual concepts with fine details.
Integrated with ChatGPT for seamless text-to-image generation within conversational context.`,
                rating: 4.8,
                users: "50M+",
                pricing: "Included with ChatGPT Plus subscription"
            },
            {
                id: 9,
                name: "PyTorch",
                category: "ml",
                license: "opensource",
                region: "Global",
                tags: ["deep-learning", "facebook", "python", "research", "tensor-computation"],
                url: "https://pytorch.org",
                logo: "resources/ai-tools/tensorflow-icon.png",
                popularity: 91,
                featured: true,
                description: `Flexible deep learning framework for research and production with dynamic computation graphs and Pythonic design.
Developed by Meta AI, provides tensor computation with GPU acceleration and deep neural network building blocks.
Features dynamic computational graphs that allow for flexible model architecture and debugging capabilities.
Includes torchvision for computer vision, torchtext for NLP, and torchaudio for audio processing tasks.
Widely used in academic research and industry for cutting-edge AI development and experimentation.`,
                rating: 4.9,
                users: "8M+",
                pricing: "Completely free and open source"
            },
            {
                id: 10,
                name: "Replicate",
                category: "ml",
                license: "freemium",
                region: "Global",
                tags: ["cloud-ml", "api", "models", "deployment", "inference"],
                url: "https://replicate.com",
                logo: "resources/ai-tools/tensorflow-icon.png",
                popularity: 82,
                featured: false,
                description: `Platform for running machine learning models in the cloud with simple API calls and extensive model library.
Enables developers to integrate state-of-the-art AI models without managing infrastructure or GPU resources.
Supports various model categories including image generation, text processing, audio synthesis, and computer vision.
Features automatic scaling, pay-per-use pricing, and one-click deployment of popular AI models.
Ideal for startups and developers building AI-powered applications without ML infrastructure expertise.`,
                rating: 4.3,
                users: "500K+",
                pricing: "Pay-per-use, free tier available"
            },
            {
                id: 11,
                name: "LangChain",
                category: "nlp",
                license: "opensource",
                region: "Global",
                tags: ["language-models", "chains", "python", "javascript", "llm-framework"],
                url: "https://langchain.com",
                logo: "resources/ai-tools/tensorflow-icon.png",
                popularity: 83,
                featured: false,
                description: `Framework for building applications powered by language models with chain-based architecture and tool integration.
Simplifies the development of LLM applications through modular components and pre-built chains for common tasks.
Supports integration with various LLM providers, vector databases, and external APIs for extended functionality.
Features include document loading, text splitting, embeddings, and retrieval-augmented generation capabilities.
Essential for developers building chatbots, question-answering systems, and AI-powered automation tools.`,
                rating: 4.5,
                users: "1M+",
                pricing: "Free and open source, cloud services available"
            },
            {
                id: 12,
                name: "Pinecone",
                category: "data",
                license: "freemium",
                region: "Global",
                tags: ["vector-database", "semantic-search", "ai-applications", "similarity-search"],
                url: "https://pinecone.io",
                logo: "resources/ai-tools/tableau-icon.png",
                popularity: 81,
                featured: false,
                description: `Vector database for building scalable AI applications with semantic search and similarity matching capabilities.
Designed specifically for high-dimensional vector embeddings used in modern AI and machine learning applications.
Provides millisecond-scale similarity search across billions of vectors with high accuracy and low latency.
Features include real-time updates, metadata filtering, and horizontal scaling for production workloads.
Essential for building recommendation systems, semantic search, and retrieval-augmented generation applications.`,
                rating: 4.4,
                users: "100K+",
                pricing: "Free tier available, paid plans for scale"
            },
            {
                id: 13,
                name: "Weights & Biases",
                category: "ml",
                license: "freemium",
                region: "Global",
                tags: ["experiment-tracking", "ml-ops", "collaboration", "model-management"],
                url: "https://wandb.ai",
                logo: "resources/ai-tools/tensorflow-icon.png",
                popularity: 80,
                featured: false,
                description: `Machine learning platform for experiment tracking, model management, and collaborative ML development.
Provides comprehensive logging of experiments, hyperparameters, metrics, and model artifacts during training.
Features interactive visualizations, model comparison, and team collaboration tools for ML projects.
Includes model registry, dataset versioning, and automated hyperparameter optimization capabilities.
Essential for ML teams managing complex experiments, model deployment, and reproducible research workflows.`,
                rating: 4.6,
                users: "200K+",
                pricing: "Free tier available, paid plans for teams"
            },
            {
                id: 14,
                name: "Gradio",
                category: "dev",
                license: "opensource",
                region: "Global",
                tags: ["interface", "python", "ml-models", "web-app", "demo"],
                url: "https://gradio.app",
                logo: "resources/ai-tools/copilot-icon.png",
                popularity: 79,
                featured: false,
                description: `Python library for creating interactive web interfaces for machine learning models and AI applications.
Enables rapid prototyping and sharing of ML models through simple, intuitive web interfaces with minimal code.
Supports various input types including text, images, audio, and video with automatic UI generation.
Features include live model demos, API generation, and easy deployment to public URLs for sharing.
Popular among researchers and developers for creating model demonstrations and collecting user feedback.`,
                rating: 4.4,
                users: "300K+",
                pricing: "Free and open source"
            },
            {
                id: 15,
                name: "OpenCV",
                category: "vision",
                license: "opensource",
                region: "Global",
                tags: ["computer-vision", "image-processing", "opencv", "c-plus-plus", "python"],
                url: "https://opencv.org",
                logo: "resources/ai-tools/midjourney-icon.png",
                popularity: 85,
                featured: false,
                description: `Comprehensive computer vision library with optimized algorithms for image processing and machine vision applications.
Provides extensive collection of computer vision algorithms including feature detection, object tracking, and camera calibration.
Supports multiple programming languages including C++, Python, Java, and JavaScript with cross-platform compatibility.
Features include real-time image processing, machine learning integration, and GPU acceleration for performance-critical applications.
Essential for computer vision research, robotics, medical imaging, and industrial automation applications.`,
                rating: 4.7,
                users: "5M+",
                pricing: "Free and open source"
            }
        ];
        
        // Enhanced Courses Database
        this.courses = [
            {
                id: 1,
                title: "AI Fundamentals",
                provider: "Stanford University",
                description: `Comprehensive introduction to artificial intelligence concepts, algorithms, and applications.
Covers fundamental principles of AI including search algorithms, knowledge representation, and machine learning basics.
Explores ethical considerations in AI development and societal impact of artificial intelligence.
Hands-on projects and real-world case studies to reinforce theoretical concepts.
Suitable for beginners with basic programming knowledge and strong mathematical foundation.`,
                difficulty: "Beginner",
                duration: "8 weeks",
                category: "fundamentals",
                free: true,
                rating: 4.8,
                students: 15420,
                thumbnail: "resources/course-images/ai-fundamentals.png",
                enrollLink: "https://coursera.org/ai-fundamentals",
                curriculum: [
                    "Introduction to Artificial Intelligence",
                    "Search Algorithms and Problem Solving",
                    "Knowledge Representation and Logic",
                    "Machine Learning Fundamentals",
                    "Neural Networks and Deep Learning",
                    "Natural Language Processing Basics",
                    "Computer Vision Introduction",
                    "AI Ethics and Future Implications"
                ],
                instructor: "Dr. Andrew Ng",
                language: "English",
                certificate: true
            },
            {
                id: 2,
                title: "Machine Learning Specialization",
                provider: "DeepLearning.AI",
                description: `Master machine learning fundamentals through hands-on projects and real-world applications.
Comprehensive coverage of supervised and unsupervised learning algorithms with practical implementations.
Explores neural network architectures, optimization techniques, and model evaluation methods.
Industry-relevant projects and case studies from leading technology companies.
Designed for learners with basic programming and mathematical background.`,
                difficulty: "Intermediate",
                duration: "12 weeks",
                category: "machine-learning",
                free: true,
                rating: 4.9,
                students: 28350,
                thumbnail: "resources/course-images/machine-learning.png",
                enrollLink: "https://coursera.org/ml-specialization",
                curriculum: [
                    "Supervised Learning Fundamentals",
                    "Advanced Learning Algorithms",
                    "Unsupervised Learning Techniques",
                    "Neural Networks and Deep Learning",
                    "Model Evaluation and Optimization",
                    "Feature Engineering and Selection",
                    "Ensemble Methods and Boosting",
                    "Real-world ML Applications"
                ],
                instructor: "Dr. Andrew Ng",
                language: "English",
                certificate: true
            },
            {
                id: 3,
                title: "Deep Learning with PyTorch",
                provider: "Facebook AI",
                description: `Build and train deep neural networks using PyTorch framework with practical examples.
Covers convolutional networks, RNNs, LSTMs, and advanced deep learning techniques.
Hands-on implementation of state-of-the-art architectures for computer vision and NLP tasks.
Real-world projects including image classification, text generation, and sequence modeling.
Requires intermediate Python programming and basic machine learning knowledge.`,
                difficulty: "Advanced",
                duration: "10 weeks",
                category: "deep-learning",
                free: true,
                rating: 4.7,
                students: 12300,
                thumbnail: "resources/course-images/deep-learning.png",
                enrollLink: "https://pytorch.org/tutorials/",
                curriculum: [
                    "PyTorch Fundamentals and Tensors",
                    "Neural Network Building Blocks",
                    "Convolutional Neural Networks",
                    "Recurrent Neural Networks and LSTMs",
                    "Transfer Learning and Fine-tuning",
                    "Generative Models and GANs",
                    "Attention Mechanisms and Transformers",
                    "Model Deployment and Optimization"
                ],
                instructor: "PyTorch Team",
                language: "English",
                certificate: true
            },
            {
                id: 4,
                title: "Natural Language Processing",
                provider: "Google AI",
                description: `Learn to process and understand human language using modern NLP techniques and models.
Covers tokenization, sentiment analysis, named entity recognition, and machine translation.
Explores transformer architectures, BERT, GPT models, and latest advances in language AI.
Practical applications including chatbots, text summarization, and question-answering systems.
Requires programming experience and basic machine learning knowledge.`,
                difficulty: "Intermediate",
                duration: "6 weeks",
                category: "nlp",
                free: true,
                rating: 4.6,
                students: 18900,
                thumbnail: "resources/course-images/ai-fundamentals.png",
                enrollLink: "https://ai.google/education/",
                curriculum: [
                    "NLP Fundamentals and Text Preprocessing",
                    "Language Modeling and Word Embeddings",
                    "Recurrent Neural Networks for Text",
                    "Attention Mechanisms and Transformers",
                    "BERT and Pre-trained Language Models",
                    "Text Classification and Sentiment Analysis",
                    "Named Entity Recognition and Information Extraction",
                    "Machine Translation and Text Generation"
                ],
                instructor: "Google AI Team",
                language: "English",
                certificate: true
            },
            {
                id: 5,
                title: "Computer Vision Basics",
                provider: "University of Buffalo",
                description: `Introduction to computer vision concepts including image processing and object detection.
Covers fundamental algorithms for feature detection, image filtering, and geometric transformations.
Explores convolutional neural networks and modern deep learning approaches for visual recognition.
Hands-on projects with real-world applications in medical imaging, autonomous vehicles, and robotics.
Suitable for beginners with basic programming and mathematical background.`,
                difficulty: "Beginner",
                duration: "4 weeks",
                category: "computer-vision",
                free: true,
                rating: 4.5,
                students: 22100,
                thumbnail: "resources/course-images/machine-learning.png",
                enrollLink: "https://coursera.org/computer-vision",
                curriculum: [
                    "Digital Image Processing Fundamentals",
                    "Image Filtering and Edge Detection",
                    "Feature Detection and Matching",
                    "Image Classification Basics",
                    "Convolutional Neural Networks",
                    "Object Detection and Localization",
                    "Image Segmentation Techniques",
                    "Applications in Medical Imaging"
                ],
                instructor: "Dr. David Smith",
                language: "English",
                certificate: true
            },
            {
                id: 6,
                title: "AI Ethics and Society",
                provider: "MIT",
                description: `Explore the ethical implications of AI technology, bias in algorithms, and societal impact.
Covers fairness in machine learning, privacy concerns, and responsible AI development practices.
Examines case studies of AI ethics failures and successful implementation of ethical AI systems.
Discusses regulatory frameworks, governance models, and future directions for AI policy.
Open to learners from all backgrounds interested in technology ethics and social impact.`,
                difficulty: "Intermediate",
                duration: "5 weeks",
                category: "ethics",
                free: true,
                rating: 4.8,
                students: 9800,
                thumbnail: "resources/course-images/ai-fundamentals.png",
                enrollLink: "https://mit.edu/ai-ethics",
                curriculum: [
                    "Introduction to AI Ethics",
                    "Bias and Fairness in Machine Learning",
                    "Privacy and Surveillance in AI Systems",
                    "Accountability and Transparency in AI",
                    "AI Governance and Regulatory Frameworks",
                    "Case Studies in AI Ethics",
                    "Designing Ethical AI Systems",
                    "Future of AI and Society"
                ],
                instructor: "Dr. Kate Crawford",
                language: "English",
                certificate: true
            }
        ];
        
        // Student Management System
        this.students = [
            {
                id: 1,
                name: "John Doe",
                email: "john.doe@email.com",
                password: "hashed_password_123",
                role: "student",
                enrolledCourses: [1, 2],
                completedCourses: [1],
                savedTools: [1, 3, 4],
                learningStreak: 7,
                totalHours: 47,
                certificates: [1],
                joinDate: "2024-01-15",
                lastActive: "2024-11-16",
                profile: {
                    bio: "AI enthusiast and machine learning practitioner",
                    interests: ["NLP", "Computer Vision", "Deep Learning"],
                    level: "Intermediate"
                }
            },
            {
                id: 2,
                name: "Jane Smith",
                email: "jane.smith@email.com",
                password: "hashed_password_456",
                role: "student",
                enrolledCourses: [2, 3, 4],
                completedCourses: [],
                savedTools: [2, 5, 6, 8],
                learningStreak: 12,
                totalHours: 89,
                certificates: [],
                joinDate: "2024-02-20",
                lastActive: "2024-11-16",
                profile: {
                    bio: "Computer science student passionate about AI",
                    interests: ["Machine Learning", "Data Science", "AI Ethics"],
                    level: "Beginner"
                }
            },
            {
                id: 3,
                name: "Mike Johnson",
                email: "mike.j@email.com",
                password: "hashed_password_789",
                role: "student",
                enrolledCourses: [1, 5],
                completedCourses: [1, 5],
                savedTools: [7, 9, 10, 11],
                learningStreak: 3,
                totalHours: 156,
                certificates: [1, 5],
                joinDate: "2023-12-10",
                lastActive: "2024-11-15",
                profile: {
                    bio: "Software engineer transitioning to AI/ML",
                    interests: ["Deep Learning", "MLOps", "Computer Vision"],
                    level: "Advanced"
                }
            },
            {
                id: 4,
                name: "Sarah Wilson",
                email: "sarah.wilson@email.com",
                password: "hashed_password_012",
                role: "student",
                enrolledCourses: [6],
                completedCourses: [],
                savedTools: [12, 13, 14],
                learningStreak: 5,
                totalHours: 23,
                certificates: [],
                joinDate: "2024-03-05",
                lastActive: "2024-11-16",
                profile: {
                    bio: "Ethics researcher exploring AI implications",
                    interests: ["AI Ethics", "Policy", "Social Impact"],
                    level: "Intermediate"
                }
            }
        ];
        
        // Enrollment Tracking System
        this.enrollments = [
            {
                id: 1,
                studentId: 1,
                courseId: 1,
                enrollmentDate: "2024-01-20",
                progress: 100,
                completedDate: "2024-03-15",
                certificateId: "CERT-2024-001",
                grade: "A"
            },
            {
                id: 2,
                studentId: 1,
                courseId: 2,
                enrollmentDate: "2024-03-20",
                progress: 75,
                completedDate: null,
                certificateId: null,
                grade: null
            },
            {
                id: 3,
                studentId: 2,
                courseId: 2,
                enrollmentDate: "2024-02-25",
                progress: 45,
                completedDate: null,
                certificateId: null,
                grade: null
            },
            {
                id: 4,
                studentId: 2,
                courseId: 3,
                enrollmentDate: "2024-04-01",
                progress: 20,
                completedDate: null,
                certificateId: null,
                grade: null
            },
            {
                id: 5,
                studentId: 2,
                courseId: 4,
                enrollmentDate: "2024-04-15",
                progress: 10,
                completedDate: null,
                certificateId: null,
                grade: null
            },
            {
                id: 6,
                studentId: 3,
                courseId: 1,
                enrollmentDate: "2023-12-15",
                progress: 100,
                completedDate: "2024-02-01",
                certificateId: "CERT-2024-002",
                grade: "A+"
            },
            {
                id: 7,
                studentId: 3,
                courseId: 5,
                enrollmentDate: "2024-02-10",
                progress: 100,
                completedDate: "2024-04-20",
                certificateId: "CERT-2024-003",
                grade: "A"
            },
            {
                id: 8,
                studentId: 4,
                courseId: 6,
                enrollmentDate: "2024-03-10",
                progress: 30,
                completedDate: null,
                certificateId: null,
                grade: null
            }
        ];
        
        // Tool Bookmarks/Saves
        this.toolBookmarks = [
            { studentId: 1, toolId: 1, savedDate: "2024-01-25" },
            { studentId: 1, toolId: 3, savedDate: "2024-02-01" },
            { studentId: 1, toolId: 4, savedDate: "2024-02-15" },
            { studentId: 2, toolId: 2, savedDate: "2024-02-20" },
            { studentId: 2, toolId: 5, savedDate: "2024-03-01" },
            { studentId: 2, toolId: 6, savedDate: "2024-03-10" },
            { studentId: 2, toolId: 8, savedDate: "2024-03-15" },
            { studentId: 3, toolId: 7, savedDate: "2024-01-10" },
            { studentId: 3, toolId: 9, savedDate: "2024-01-20" },
            { studentId: 3, toolId: 10, savedDate: "2024-02-05" },
            { studentId: 3, toolId: 11, savedDate: "2024-02-20" },
            { studentId: 4, toolId: 12, savedDate: "2024-03-20" },
            { studentId: 4, toolId: 13, savedDate: "2024-03-25" },
            { studentId: 4, toolId: 14, savedDate: "2024-04-01" }
        ];
        
        // Learning Progress Tracking
        this.learningProgress = [
            {
                studentId: 1,
                courseId: 2,
                lessonsCompleted: 6,
                totalLessons: 8,
                lastAccessed: "2024-11-16",
                timeSpent: 1800, // minutes
                currentLesson: "Neural Networks and Deep Learning",
                assignmentsSubmitted: 5,
                quizScores: [85, 92, 78, 88, 90]
            },
            {
                studentId: 2,
                courseId: 2,
                lessonsCompleted: 3,
                totalLessons: 8,
                lastAccessed: "2024-11-15",
                timeSpent: 1200,
                currentLesson: "Advanced Learning Algorithms",
                assignmentsSubmitted: 2,
                quizScores: [82, 79]
            },
            {
                studentId: 2,
                courseId: 3,
                lessonsCompleted: 1,
                totalLessons: 8,
                lastAccessed: "2024-11-14",
                timeSpent: 300,
                currentLesson: "PyTorch Fundamentals and Tensors",
                assignmentsSubmitted: 0,
                quizScores: []
            }
        ];
    }
    
    // Student Management Methods
    getStudentById(id) {
        return this.students.find(student => student.id === id);
    }
    
    getStudentByEmail(email) {
        return this.students.find(student => student.email === email);
    }
    
    createStudent(studentData) {
        const newStudent = {
            id: this.students.length + 1,
            ...studentData,
            role: "student",
            enrolledCourses: [],
            completedCourses: [],
            savedTools: [],
            learningStreak: 0,
            totalHours: 0,
            certificates: [],
            joinDate: new Date().toISOString().split('T')[0],
            lastActive: new Date().toISOString().split('T')[0],
            profile: {
                bio: studentData.bio || "",
                interests: studentData.interests || [],
                level: studentData.level || "Beginner"
            }
        };
        
        this.students.push(newStudent);
        return newStudent;
    }
    
    enrollStudentInCourse(studentId, courseId) {
        const existingEnrollment = this.enrollments.find(
            e => e.studentId === studentId && e.courseId === courseId
        );
        
        if (existingEnrollment) {
            return { success: false, message: "Already enrolled in this course" };
        }
        
        const newEnrollment = {
            id: this.enrollments.length + 1,
            studentId,
            courseId,
            enrollmentDate: new Date().toISOString().split('T')[0],
            progress: 0,
            completedDate: null,
            certificateId: null,
            grade: null
        };
        
        this.enrollments.push(newEnrollment);
        
        // Update student's enrolled courses
        const student = this.getStudentById(studentId);
        if (student && !student.enrolledCourses.includes(courseId)) {
            student.enrolledCourses.push(courseId);
        }
        
        return { success: true, enrollment: newEnrollment };
    }
    
    completeCourse(studentId, courseId, grade = "A") {
        const enrollment = this.enrollments.find(
            e => e.studentId === studentId && e.courseId === courseId
        );
        
        if (!enrollment) {
            return { success: false, message: "Not enrolled in this course" };
        }
        
        enrollment.progress = 100;
        enrollment.completedDate = new Date().toISOString().split('T')[0];
        enrollment.certificateId = `CERT-${new Date().getFullYear()}-${String(this.enrollments.length).padStart(3, '0')}`;
        enrollment.grade = grade;
        
        // Update student's completed courses
        const student = this.getStudentById(studentId);
        if (student) {
            student.completedCourses.push(courseId);
            student.certificates.push(enrollment.certificateId);
        }
        
        return { success: true, certificateId: enrollment.certificateId };
    }
    
    saveToolForStudent(studentId, toolId) {
        const existingBookmark = this.toolBookmarks.find(
            b => b.studentId === studentId && b.toolId === toolId
        );
        
        if (existingBookmark) {
            return { success: false, message: "Tool already saved" };
        }
        
        const newBookmark = {
            studentId,
            toolId,
            savedDate: new Date().toISOString().split('T')[0]
        };
        
        this.toolBookmarks.push(newBookmark);
        
        // Update student's saved tools
        const student = this.getStudentById(studentId);
        if (student && !student.savedTools.includes(toolId)) {
            student.savedTools.push(toolId);
        }
        
        return { success: true };
    }
    
    removeSavedTool(studentId, toolId) {
        const bookmarkIndex = this.toolBookmarks.findIndex(
            b => b.studentId === studentId && b.toolId === toolId
        );
        
        if (bookmarkIndex === -1) {
            return { success: false, message: "Tool not found in bookmarks" };
        }
        
        this.toolBookmarks.splice(bookmarkIndex, 1);
        
        // Update student's saved tools
        const student = this.getStudentById(studentId);
        if (student) {
            const toolIndex = student.savedTools.indexOf(toolId);
            if (toolIndex > -1) {
                student.savedTools.splice(toolIndex, 1);
            }
        }
        
        return { success: true };
    }
    
    getStudentEnrollments(studentId) {
        return this.enrollments.filter(e => e.studentId === studentId);
    }
    
    getStudentProgress(studentId, courseId) {
        return this.learningProgress.find(
            p => p.studentId === studentId && p.courseId === courseId
        );
    }
    
    updateLearningProgress(studentId, courseId, progressData) {
        let progress = this.learningProgress.find(
            p => p.studentId === studentId && p.courseId === courseId
        );
        
        if (!progress) {
            progress = {
                studentId,
                courseId,
                lessonsCompleted: 0,
                totalLessons: progressData.totalLessons || 8,
                lastAccessed: new Date().toISOString().split('T')[0],
                timeSpent: 0,
                currentLesson: "",
                assignmentsSubmitted: 0,
                quizScores: []
            };
            this.learningProgress.push(progress);
        }
        
        // Update progress data
        Object.assign(progress, progressData);
        progress.lastAccessed = new Date().toISOString().split('T')[0];
        
        return progress;
    }
    
    // Tool Management Methods
    addTool(toolData) {
        const newTool = {
            id: this.tools.length + 1,
            ...toolData,
            popularity: toolData.popularity || 50,
            featured: toolData.featured || false,
            rating: toolData.rating || 4.0,
            users: toolData.users || "1K+",
            pricing: toolData.pricing || "Free"
        };
        
        this.tools.push(newTool);
        return newTool;
    }
    
    updateTool(id, toolData) {
        const toolIndex = this.tools.findIndex(t => t.id === id);
        if (toolIndex === -1) {
            return { success: false, message: "Tool not found" };
        }
        
        this.tools[toolIndex] = { ...this.tools[toolIndex], ...toolData };
        return { success: true, tool: this.tools[toolIndex] };
    }
    
    deleteTool(id) {
        const toolIndex = this.tools.findIndex(t => t.id === id);
        if (toolIndex === -1) {
            return { success: false, message: "Tool not found" };
        }
        
        this.tools.splice(toolIndex, 1);
        return { success: true };
    }
    
    getToolsByCategory(category) {
        return this.tools.filter(tool => tool.category === category);
    }
    
    searchTools(query) {
        const searchTerm = query.toLowerCase();
        return this.tools.filter(tool => 
            tool.name.toLowerCase().includes(searchTerm) ||
            tool.description.toLowerCase().includes(searchTerm) ||
            tool.tags.some(tag => tag.toLowerCase().includes(searchTerm))
        );
    }
    
    // Course Management Methods
    addCourse(courseData) {
        const newCourse = {
            id: this.courses.length + 1,
            ...courseData,
            students: 0,
            rating: courseData.rating || 4.0,
            certificate: courseData.certificate || true
        };
        
        this.courses.push(newCourse);
        return newCourse;
    }
    
    updateCourse(id, courseData) {
        const courseIndex = this.courses.findIndex(c => c.id === id);
        if (courseIndex === -1) {
            return { success: false, message: "Course not found" };
        }
        
        this.courses[courseIndex] = { ...this.courses[courseIndex], ...courseData };
        return { success: true, course: this.courses[courseIndex] };
    }
    
    deleteCourse(id) {
        const courseIndex = this.courses.findIndex(c => c.id === id);
        if (courseIndex === -1) {
            return { success: false, message: "Course not found" };
        }
        
        this.courses.splice(courseIndex, 1);
        return { success: true };
    }
    
    // Analytics Methods
    getPlatformStats() {
        return {
            totalTools: this.tools.length,
            totalCourses: this.courses.length,
            totalStudents: this.students.length,
            totalEnrollments: this.enrollments.length,
            completedEnrollments: this.enrollments.filter(e => e.progress === 100).length,
            activeStudentsToday: this.students.filter(s => s.lastActive === new Date().toISOString().split('T')[0]).length,
            averageRating: this.tools.reduce((sum, tool) => sum + tool.rating, 0) / this.tools.length
        };
    }
    
    getStudentAnalytics(studentId) {
        const student = this.getStudentById(studentId);
        if (!student) return null;
        
        const enrollments = this.getStudentEnrollments(studentId);
        const completedCount = enrollments.filter(e => e.progress === 100).length;
        const totalTimeSpent = this.learningProgress
            .filter(p => p.studentId === studentId)
            .reduce((sum, p) => sum + p.timeSpent, 0);
        
        return {
            enrolledCourses: enrollments.length,
            completedCourses: completedCount,
            savedTools: student.savedTools.length,
            learningStreak: student.learningStreak,
            totalHours: Math.round(totalTimeSpent / 60),
            certificatesEarned: student.certificates.length,
            averageProgress: enrollments.length > 0 ? 
                enrollments.reduce((sum, e) => sum + e.progress, 0) / enrollments.length : 0
        };
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = NanoLancerDatabase;
} else {
    window.NanoLancerDatabase = NanoLancerDatabase;
}