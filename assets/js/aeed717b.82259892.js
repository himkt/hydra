"use strict";(self.webpackChunkwebsite=self.webpackChunkwebsite||[]).push([[6457],{3905:function(e,t,n){n.d(t,{Zo:function(){return u},kt:function(){return d}});var r=n(7294);function i(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function a(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,r)}return n}function o(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?a(Object(n),!0).forEach((function(t){i(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):a(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function l(e,t){if(null==e)return{};var n,r,i=function(e,t){if(null==e)return{};var n,r,i={},a=Object.keys(e);for(r=0;r<a.length;r++)n=a[r],t.indexOf(n)>=0||(i[n]=e[n]);return i}(e,t);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);for(r=0;r<a.length;r++)n=a[r],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(i[n]=e[n])}return i}var c=r.createContext({}),p=function(e){var t=r.useContext(c),n=t;return e&&(n="function"==typeof e?e(t):o(o({},t),e)),n},u=function(e){var t=p(e.components);return r.createElement(c.Provider,{value:t},e.children)},s={inlineCode:"code",wrapper:function(e){var t=e.children;return r.createElement(r.Fragment,{},t)}},m=r.forwardRef((function(e,t){var n=e.components,i=e.mdxType,a=e.originalType,c=e.parentName,u=l(e,["components","mdxType","originalType","parentName"]),m=p(n),d=i,y=m["".concat(c,".").concat(d)]||m[d]||s[d]||a;return n?r.createElement(y,o(o({ref:t},u),{},{components:n})):r.createElement(y,o({ref:t},u))}));function d(e,t){var n=arguments,i=t&&t.mdxType;if("string"==typeof e||i){var a=n.length,o=new Array(a);o[0]=m;var l={};for(var c in t)hasOwnProperty.call(t,c)&&(l[c]=t[c]);l.originalType=e,l.mdxType="string"==typeof e?e:i,o[1]=l;for(var p=2;p<a;p++)o[p]=n[p];return r.createElement.apply(null,o)}return r.createElement.apply(null,n)}m.displayName="MDXCreateElement"},3899:function(e,t,n){n.d(t,{Z:function(){return c},T:function(){return p}});var r=n(7462),i=n(7294),a=n(6742),o=n(2263),l=n(907);function c(e){return i.createElement(a.Z,(0,r.Z)({},e,{to:(t=e.to,c=(0,l.zu)(),(0,o.default)().siteConfig.customFields.githubLinkVersionToBaseUrl[null!=(n=null==c?void 0:c.name)?n:"current"]+t),target:"_blank"}));var t,n,c}function p(e){var t,n=null!=(t=e.text)?t:"Example";return i.createElement(c,e,i.createElement("span",null,"\xa0"),i.createElement("img",{src:"https://img.shields.io/badge/-"+n+"-informational",alt:"Example"}))}},6436:function(e,t,n){n.r(t),n.d(t,{frontMatter:function(){return c},contentTitle:function(){return p},metadata:function(){return u},toc:function(){return s},default:function(){return d}});var r=n(7462),i=n(3366),a=(n(7294),n(3905)),o=n(3899),l=["components"],c={id:"minimal_example",title:"Minimal example"},p=void 0,u={unversionedId:"tutorials/structured_config/minimal_example",id:"tutorials/structured_config/minimal_example",isDocsHomePage:!1,title:"Minimal example",description:"There are four key elements in this example:",source:"@site/docs/tutorials/structured_config/1_minimal_example.md",sourceDirName:"tutorials/structured_config",slug:"/tutorials/structured_config/minimal_example",permalink:"/docs/next/tutorials/structured_config/minimal_example",editUrl:"https://github.com/facebookresearch/hydra/edit/main/website/docs/tutorials/structured_config/1_minimal_example.md",tags:[],version:"current",lastUpdatedBy:"Jieru Hu",lastUpdatedAt:1637025515,formattedLastUpdatedAt:"11/16/2021",sidebarPosition:1,frontMatter:{id:"minimal_example",title:"Minimal example"},sidebar:"docs",previous:{title:"Introduction to Structured Configs",permalink:"/docs/next/tutorials/structured_config/intro"},next:{title:"A hierarchical static configuration",permalink:"/docs/next/tutorials/structured_config/hierarchical_static_config"}},s=[{value:"Duck-typing enables static type checking",id:"duck-typing-enables-static-type-checking",children:[]},{value:"Structured Configs enable Hydra to catch type errors at runtime",id:"structured-configs-enable-hydra-to-catch-type-errors-at-runtime",children:[]},{value:"Duck typing",id:"duck-typing",children:[]}],m={toc:s};function d(e){var t=e.components,n=(0,i.Z)(e,l);return(0,a.kt)("wrapper",(0,r.Z)({},m,n,{components:t,mdxType:"MDXLayout"}),(0,a.kt)(o.T,{to:"examples/tutorials/structured_configs/1_minimal",mdxType:"ExampleGithubLink"}),(0,a.kt)("p",null,"There are four key elements in this example:"),(0,a.kt)("ul",null,(0,a.kt)("li",{parentName:"ul"},"A ",(0,a.kt)("inlineCode",{parentName:"li"},"@dataclass")," describes the application's configuration"),(0,a.kt)("li",{parentName:"ul"},(0,a.kt)("inlineCode",{parentName:"li"},"ConfigStore")," manages the Structured Config"),(0,a.kt)("li",{parentName:"ul"},(0,a.kt)("inlineCode",{parentName:"li"},"cfg")," is ",(0,a.kt)("inlineCode",{parentName:"li"},"duck typed")," as a ",(0,a.kt)("inlineCode",{parentName:"li"},"MySQLConfig")," instead of a ",(0,a.kt)("inlineCode",{parentName:"li"},"DictConfig")),(0,a.kt)("li",{parentName:"ul"},"There is a subtle typo in the code below, can you spot it?")),(0,a.kt)("p",null,"In this example, the config node stored in the ",(0,a.kt)("inlineCode",{parentName:"p"},"ConfigStore")," replaces the traditional ",(0,a.kt)("inlineCode",{parentName:"p"},"config.yaml")," file."),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-python",metastring:'title="my_app_type_error.py" {18}',title:'"my_app_type_error.py"',"{18}":!0},'from dataclasses import dataclass\n\nimport hydra\nfrom hydra.core.config_store import ConfigStore\n\n@dataclass\nclass MySQLConfig:\n    host: str = "localhost"\n    port: int = 3306\n\ncs = ConfigStore.instance()\n# Registering the Config class with the name \'config\'.\ncs.store(name="config", node=MySQLConfig)\n\n@hydra.main(config_path=None, config_name="config")\ndef my_app(cfg: MySQLConfig) -> None:\n    # pork should be port!\n    if cfg.pork == 80:\n        print("Is this a webserver?!")\n\nif __name__ == "__main__":\n    my_app()\n')),(0,a.kt)("h3",{id:"duck-typing-enables-static-type-checking"},"Duck-typing enables static type checking"),(0,a.kt)("p",null,"Duck-typing the config object as ",(0,a.kt)("inlineCode",{parentName:"p"},"MySQLConfig")," enables static type checkers like ",(0,a.kt)("inlineCode",{parentName:"p"},"mypy")," to catch\ntype errors before you run your code:"),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-text",metastring:'title="$ mypy my_app_type_error.py"',title:'"$',mypy:!0,'my_app_type_error.py"':!0},'my_app_type_error.py:22: error: "MySQLConfig" has no attribute "pork"\nFound 1 error in 1 file (checked 1 source file)\n')),(0,a.kt)("h3",{id:"structured-configs-enable-hydra-to-catch-type-errors-at-runtime"},"Structured Configs enable Hydra to catch type errors at runtime"),(0,a.kt)("p",null,"If you forget to run ",(0,a.kt)("inlineCode",{parentName:"p"},"mypy"),", Hydra will report the error at runtime:"),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-text",metastring:'title="$ python my_app_type_error.py"',title:'"$',python:!0,'my_app_type_error.py"':!0},"Traceback (most recent call last):\n  File \"my_app_type_error.py\", line 22, in my_app\n    if cfg.pork == 80:\nomegaconf.errors.ConfigAttributeError: Key 'pork' not in 'MySQLConfig'\n        full_key: pork\n        object_type=MySQLConfig\n\nSet the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.\n")),(0,a.kt)("p",null,"Hydra will also catch typos, or type errors in the command line:"),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre"},"$ python my_app_type_error.py port=fail\nError merging override port=fail\nValue 'fail' could not be converted to Integer\n        full_key: port\n        object_type=MySQLConfig\n")),(0,a.kt)("p",null,"We will see additional types of runtime errors that Hydra can catch later in this tutorial. Such as:"),(0,a.kt)("ul",null,(0,a.kt)("li",{parentName:"ul"},"Trying to read or write a non existent field in your config object"),(0,a.kt)("li",{parentName:"ul"},"Assigning a value that is incompatible with the declared type"),(0,a.kt)("li",{parentName:"ul"},"Attempting to modify a ",(0,a.kt)("a",{parentName:"li",href:"https://omegaconf.readthedocs.io/en/latest/structured_config.html#frozen"},"frozen config"))),(0,a.kt)("h2",{id:"duck-typing"},"Duck typing"),(0,a.kt)("p",null,"In the example above ",(0,a.kt)("inlineCode",{parentName:"p"},"cfg")," is duck typed as ",(0,a.kt)("inlineCode",{parentName:"p"},"MySQLConfig"),".\nIt is actually an instance of ",(0,a.kt)("inlineCode",{parentName:"p"},"DictConfig"),". The duck typing enables static type checking by tools like Mypy or PyCharm.\nThis reduces development time by catching coding errors before you run your application."),(0,a.kt)("p",null,"The name ",(0,a.kt)("a",{parentName:"p",href:"https://en.wikipedia.org/wiki/Duck_typing"},"Duck typing"),' comes from the phrase "If it walks like a duck, swims like a duck, and quacks like a duck, then it probably is a duck".\nIt can be useful when you care about the methods or attributes of an object, not the actual type of the object.'))}d.isMDXComponent=!0}}]);