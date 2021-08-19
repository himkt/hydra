(self.webpackChunkwebsite=self.webpackChunkwebsite||[]).push([[8062],{3905:function(e,t,n){"use strict";n.d(t,{Zo:function(){return u},kt:function(){return m}});var r=n(7294);function i(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function o(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,r)}return n}function a(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?o(Object(n),!0).forEach((function(t){i(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):o(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function s(e,t){if(null==e)return{};var n,r,i=function(e,t){if(null==e)return{};var n,r,i={},o=Object.keys(e);for(r=0;r<o.length;r++)n=o[r],t.indexOf(n)>=0||(i[n]=e[n]);return i}(e,t);if(Object.getOwnPropertySymbols){var o=Object.getOwnPropertySymbols(e);for(r=0;r<o.length;r++)n=o[r],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(i[n]=e[n])}return i}var l=r.createContext({}),p=function(e){var t=r.useContext(l),n=t;return e&&(n="function"==typeof e?e(t):a(a({},t),e)),n},u=function(e){var t=p(e.components);return r.createElement(l.Provider,{value:t},e.children)},c={inlineCode:"code",wrapper:function(e){var t=e.children;return r.createElement(r.Fragment,{},t)}},d=r.forwardRef((function(e,t){var n=e.components,i=e.mdxType,o=e.originalType,l=e.parentName,u=s(e,["components","mdxType","originalType","parentName"]),d=p(n),m=i,g=d["".concat(l,".").concat(m)]||d[m]||c[m]||o;return n?r.createElement(g,a(a({ref:t},u),{},{components:n})):r.createElement(g,a({ref:t},u))}));function m(e,t){var n=arguments,i=t&&t.mdxType;if("string"==typeof e||i){var o=n.length,a=new Array(o);a[0]=d;var s={};for(var l in t)hasOwnProperty.call(t,l)&&(s[l]=t[l]);s.originalType=e,s.mdxType="string"==typeof e?e:i,a[1]=s;for(var p=2;p<o;p++)a[p]=n[p];return r.createElement.apply(null,a)}return r.createElement.apply(null,n)}d.displayName="MDXCreateElement"},4315:function(e,t,n){"use strict";n.r(t),n.d(t,{frontMatter:function(){return s},contentTitle:function(){return l},metadata:function(){return p},toc:function(){return u},default:function(){return d}});var r=n(2122),i=n(9756),o=(n(7294),n(3905)),a=["components"],s={id:"testing",title:"Testing",sidebar_label:"Testing"},l=void 0,p={unversionedId:"development/testing",id:"development/testing",isDocsHomePage:!1,title:"Testing",description:"Hydra uses nox - a build automation tool - to manage tests, linting, code coverage, etc.",source:"@site/docs/development/testing.md",sourceDirName:"development",slug:"/development/testing",permalink:"/docs/next/development/testing",editUrl:"https://github.com/facebookresearch/hydra/edit/master/website/docs/development/testing.md",version:"current",lastUpdatedBy:"Jieru Hu",lastUpdatedAt:1629405728,formattedLastUpdatedAt:"8/19/2021",frontMatter:{id:"testing",title:"Testing",sidebar_label:"Testing"},sidebar:"docs",previous:{title:"Developer Guide Overview",permalink:"/docs/next/development/overview"},next:{title:"Style Guide",permalink:"/docs/next/development/style_guide"}},u=[{value:"Testing with pytest",id:"testing-with-pytest",children:[]},{value:"Testing with nox",id:"testing-with-nox",children:[]}],c={toc:u};function d(e){var t=e.components,n=(0,i.Z)(e,a);return(0,o.kt)("wrapper",(0,r.Z)({},c,n,{components:t,mdxType:"MDXLayout"}),(0,o.kt)("p",null,"Hydra uses ",(0,o.kt)("a",{parentName:"p",href:"https://github.com/theacodes/nox"},"nox")," - a build automation tool - to manage tests, linting, code coverage, etc.\nThe command ",(0,o.kt)("inlineCode",{parentName:"p"},"nox")," will run all the configured sessions. List the sessions using ",(0,o.kt)("inlineCode",{parentName:"p"},"nox -l")," and\nrun specific sessions with ",(0,o.kt)("inlineCode",{parentName:"p"},"nox -s NAME")," (you may need to quote the session name in some cases)"),(0,o.kt)("h2",{id:"testing-with-pytest"},"Testing with pytest"),(0,o.kt)("p",null,"Run ",(0,o.kt)("inlineCode",{parentName:"p"},"pytest")," at the repository root to run all the Hydra core tests.\nTo run the tests of individual plugins, use ",(0,o.kt)("inlineCode",{parentName:"p"},"pytest plugins/NAME")," (The plugin must be installed)."),(0,o.kt)("div",{className:"admonition admonition-info alert alert--info"},(0,o.kt)("div",{parentName:"div",className:"admonition-heading"},(0,o.kt)("h5",{parentName:"div"},(0,o.kt)("span",{parentName:"h5",className:"admonition-icon"},(0,o.kt)("svg",{parentName:"span",xmlns:"http://www.w3.org/2000/svg",width:"14",height:"16",viewBox:"0 0 14 16"},(0,o.kt)("path",{parentName:"svg",fillRule:"evenodd",d:"M7 2.3c3.14 0 5.7 2.56 5.7 5.7s-2.56 5.7-5.7 5.7A5.71 5.71 0 0 1 1.3 8c0-3.14 2.56-5.7 5.7-5.7zM7 1C3.14 1 0 4.14 0 8s3.14 7 7 7 7-3.14 7-7-3.14-7-7-7zm1 3H6v5h2V4zm0 6H6v2h2v-2z"}))),"NOTE")),(0,o.kt)("div",{parentName:"div",className:"admonition-content"},(0,o.kt)("p",{parentName:"div"},"Some plugins support fewer versions of Python than the Hydra core."))),(0,o.kt)("h2",{id:"testing-with-nox"},"Testing with nox"),(0,o.kt)("p",null,"See ",(0,o.kt)("inlineCode",{parentName:"p"},"nox -l"),". a few examples:"),(0,o.kt)("ul",null,(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("inlineCode",{parentName:"li"},"nox -s test_core")," will test Hydra core on all supported Python versions"),(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("inlineCode",{parentName:"li"},'nox -s "test_plugins-3.8"')," will test plugins on Python 3.8."),(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("inlineCode",{parentName:"li"},'nox -s "test_plugins-3.8"')," will test plugins on Python 3.8.")),(0,o.kt)("p",null,"The ",(0,o.kt)("inlineCode",{parentName:"p"},"noxfile.py")," is checking some environment variables to decide what to run. For example,\nto test a single plugin:"),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-shell",metastring:"{4}","{4}":!0},"$ PLUGINS=hydra_colorlog nox -s test_plugins-3.8\nOperating system        :       Linux\nNOX_PYTHON_VERSIONS     :       ['3.6', '3.7', '3.8', '3.9']\nPLUGINS                 :       ['hydra_colorlog']\nSKIP_CORE_TESTS         :       False\nFIX                     :       False\nVERBOSE                 :       0\nINSTALL_EDITABLE_MODE   :       0\nnox > Running session test_plugins-3.8\n...\n")))}d.isMDXComponent=!0}}]);