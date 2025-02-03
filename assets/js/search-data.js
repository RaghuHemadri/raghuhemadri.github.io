// get the ninja-keys element
const ninja = document.querySelector('ninja-keys');

// add the home and posts menu items
ninja.data = [{
    id: "nav-about",
    title: "about",
    section: "Navigation",
    handler: () => {
      window.location.href = "/";
    },
  },{id: "nav-blog",
          title: "blog",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/blog/";
          },
        },{id: "nav-publications",
          title: "publications",
          description: "publications by categories in reversed chronological order.",
          section: "Navigation",
          handler: () => {
            window.location.href = "/publications/";
          },
        },{id: "nav-projects",
          title: "projects",
          description: "A growing collection of your cool projects.",
          section: "Navigation",
          handler: () => {
            window.location.href = "/projects/";
          },
        },{id: "nav-repositories",
          title: "repositories",
          description: "Edit the `_data/repositories.yml` and change the `github_users` and `github_repos` lists to include your own GitHub profile and repositories.",
          section: "Navigation",
          handler: () => {
            window.location.href = "/repositories/";
          },
        },{id: "nav-cv",
          title: "cv",
          description: "TL;DR. This page contains my full resume. Click the PDF icon for a condensed one-page version.",
          section: "Navigation",
          handler: () => {
            window.location.href = "/cv/";
          },
        },{id: "nav-teaching",
          title: "teaching",
          description: "Materials for courses you taught. Replace this text with your description.",
          section: "Navigation",
          handler: () => {
            window.location.href = "/teaching/";
          },
        },{id: "nav-people",
          title: "people",
          description: "members of the lab or group",
          section: "Navigation",
          handler: () => {
            window.location.href = "/people/";
          },
        },{id: "dropdown-publications",
              title: "publications",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "";
              },
            },{id: "dropdown-projects",
              title: "projects",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "";
              },
            },{id: "dropdown-blog",
              title: "blog",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "/blog/";
              },
            },{id: "post-mlops-cloud-computing",
      
        title: "MLOps: Cloud Computing",
      
      description: "Cloud Computing Basics for MLOps",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2025/mlops-cloud/";
        
      },
    },{id: "post-mlops-cloud-computing-definitions",
      
        title: "MLOps: Cloud Computing Definitions",
      
      description: "Detailed explanations of concepts from `MLOps: Cloud Computing` blog",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2025/mlops-cloud-def/";
        
      },
    },{id: "post-low-level-design-a-comprehensive-guide",
      
        title: "Low-Level Design: A Comprehensive Guide",
      
      description: "Explores fundamental principles, best practices, and patterns for crafting efficient, scalable, and maintainable system architectures.",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2025/lld/";
        
      },
    },{id: "post-mlops-introduction",
      
        title: "MLOps: Introduction",
      
      description: "Introduction to MLOps and discussion on why it is important.",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2025/mlops-intro/";
        
      },
    },{id: "post-mlops-overview",
      
        title: "MLOps: Overview",
      
      description: "Overview of the blog series",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2025/mlops-overview/";
        
      },
    },{id: "post-google-gemini-updates-flash-1-5-gemma-2-and-project-astra",
      
        title: 'Google Gemini updates: Flash 1.5, Gemma 2 and Project Astra <svg width="1.2rem" height="1.2rem" top=".5rem" viewBox="0 0 40 40" xmlns="http://www.w3.org/2000/svg"><path d="M17 13.5v6H5v-12h6m3-3h6v6m0-6-9 9" class="icon_svg-stroke" stroke="#999" stroke-width="1.5" fill="none" fill-rule="evenodd" stroke-linecap="round" stroke-linejoin="round"></path></svg>',
      
      description: "We’re sharing updates across our Gemini family of models and a glimpse of Project Astra, our vision for the future of AI assistants.",
      section: "Posts",
      handler: () => {
        
          window.open("https://blog.google/technology/ai/google-gemini-update-flash-ai-assistant-io-2024/", "_blank");
        
      },
    },{id: "post-displaying-external-posts-on-your-al-folio-blog",
      
        title: 'Displaying External Posts on Your al-folio Blog <svg width="1.2rem" height="1.2rem" top=".5rem" viewBox="0 0 40 40" xmlns="http://www.w3.org/2000/svg"><path d="M17 13.5v6H5v-12h6m3-3h6v6m0-6-9 9" class="icon_svg-stroke" stroke="#999" stroke-width="1.5" fill="none" fill-rule="evenodd" stroke-linecap="round" stroke-linejoin="round"></path></svg>',
      
      description: "",
      section: "Posts",
      handler: () => {
        
          window.open("https://medium.com/@al-folio/displaying-external-posts-on-your-al-folio-blog-b60a1d241a0a?source=rss-17feae71c3c4------2", "_blank");
        
      },
    },{id: "news-a-simple-inline-announcement",
          title: 'A simple inline announcement.',
          description: "",
          section: "News",},{id: "news-a-long-announcement-with-details",
          title: 'A long announcement with details',
          description: "",
          section: "News",handler: () => {
              window.location.href = "/news/announcement_2/";
            },},{id: "news-a-simple-inline-announcement-with-markdown-emoji-sparkles-smile",
          title: 'A simple inline announcement with Markdown emoji! :sparkles: :smile:',
          description: "",
          section: "News",},{id: "projects-social-neuro-rl",
          title: 'Social Neuro RL',
          description: "Explores context-conditioned reasoning and compositional thinking in AI agents using MARL to enhance coordination, generalization, and cognitive architectures inspired by social neuroscience.",
          section: "Projects",handler: () => {
              window.location.href = "/projects/10_project/";
            },},{id: "projects-magnet-multi-agent-generative-network",
          title: 'MAGNET: Multi-Agent Generative Network',
          description: "Multi-Agent Generative Network, a dual-agent cooperative architecture for data generation, outperforming traditional GANs in producing high-quality images resembling the CelebA dataset.",
          section: "Projects",handler: () => {
              window.location.href = "/projects/11_project/";
            },},{id: "projects-meta-learning-with-exact-information-estimation",
          title: 'Meta Learning with exact information estimation',
          description: "An information-theoretic meta-regularizer to mitigate overfitting in meta-learning by maximizing mutual information using MINE, improving generalization on non-mutually-exclusive tasks.",
          section: "Projects",handler: () => {
              window.location.href = "/projects/12_project/";
            },},{id: "projects-age-and-gender-estimation",
          title: 'Age and gender estimation',
          description: "A CNN-based system using TensorFlow and dlib for accurate age and gender estimation from facial images, enabling multi-face demographic analysis.",
          section: "Projects",handler: () => {
              window.location.href = "/projects/1_project/";
            },},{id: "projects-rally-master-ai",
          title: 'Rally Master AI',
          description: "An AI-driven badminton training analysis system using perspective transformation and bird’s-eye view analysis to provide data-driven posture correction and performance assessment.",
          section: "Projects",handler: () => {
              window.location.href = "/projects/2_project/";
            },},{id: "projects-covivac-bot",
          title: 'Covivac Bot',
          description: "A telegram bot for real-time vaccine slot availability checks and instant alerts, enabling seamless and efficient appointment booking.",
          section: "Projects",handler: () => {
              window.location.href = "/projects/3_project/";
            },},{id: "projects-log-file-summarization",
          title: 'Log File Summarization',
          description: "A log file summarization tool using the Drain algorithm to extract key insights and patterns, improving system monitoring and troubleshooting efficiency.",
          section: "Projects",handler: () => {
              window.location.href = "/projects/4_project/";
            },},{id: "projects-3d-scene-reconstruction",
          title: '3D Scene Reconstruction',
          description: "Accurate 3D maps using for multi-view scene reconstruction.",
          section: "Projects",handler: () => {
              window.location.href = "/projects/5_project/";
            },},{id: "projects-fitgen-personalized-exercise-planner",
          title: 'FitGen - Personalized Exercise Planner',
          description: "Generate personalized exercise plans, optimizing workouts through real-time form feedback to enhance fitness efficiency and reduce injury risk.",
          section: "Projects",handler: () => {
              window.location.href = "/projects/6_project/";
            },},{id: "projects-reinforcement-learning-with-unity",
          title: 'Reinforcement Learning with Unity',
          description: "A RL project using Unity ML Agents, training intelligent agents in virtual environments for applications in gaming, robotics, and simulations.",
          section: "Projects",handler: () => {
              window.location.href = "/projects/7_project/";
            },},{id: "projects-pytorch-powered-machine-learning-models",
          title: 'PyTorch-Powered Machine Learning Models',
          description: "A comprehensive repository of machine learning models in PyTorch, providing well-structured implementations for applications in computer vision, NLP, and beyond.",
          section: "Projects",handler: () => {
              window.location.href = "/projects/8_project/";
            },},{id: "projects-stock-trading-using-rl",
          title: 'Stock Trading using RL',
          description: "A RL-based stock trading system, training RL agents on historical data to optimize portfolio management and enhance market returns.",
          section: "Projects",handler: () => {
              window.location.href = "/projects/9_project/";
            },},{
        id: 'social-email',
        title: 'email',
        section: 'Socials',
        handler: () => {
          window.open("mailto:%72%61%67%68%75.%68%65%6D%61%64%72%69@%6E%79%75.%65%64%75", "_blank");
        },
      },{
        id: 'social-github',
        title: 'GitHub',
        section: 'Socials',
        handler: () => {
          window.open("https://github.com/RaghuHemadri", "_blank");
        },
      },{
        id: 'social-leetcode',
        title: 'LeetCode',
        section: 'Socials',
        handler: () => {
          window.open("https://leetcode.com/u/rvhemadri/", "_blank");
        },
      },{
        id: 'social-linkedin',
        title: 'LinkedIn',
        section: 'Socials',
        handler: () => {
          window.open("https://www.linkedin.com/in/raghuhemadri", "_blank");
        },
      },{
        id: 'social-rss',
        title: 'RSS Feed',
        section: 'Socials',
        handler: () => {
          window.open("/feed.xml", "_blank");
        },
      },{
        id: 'social-scholar',
        title: 'Google Scholar',
        section: 'Socials',
        handler: () => {
          window.open("https://scholar.google.com/citations?user=eb3JyOoAAAAJ", "_blank");
        },
      },{
      id: 'light-theme',
      title: 'Change theme to light',
      description: 'Change the theme of the site to Light',
      section: 'Theme',
      handler: () => {
        setThemeSetting("light");
      },
    },
    {
      id: 'dark-theme',
      title: 'Change theme to dark',
      description: 'Change the theme of the site to Dark',
      section: 'Theme',
      handler: () => {
        setThemeSetting("dark");
      },
    },
    {
      id: 'system-theme',
      title: 'Use system default theme',
      description: 'Change the theme of the site to System Default',
      section: 'Theme',
      handler: () => {
        setThemeSetting("system");
      },
    },];
